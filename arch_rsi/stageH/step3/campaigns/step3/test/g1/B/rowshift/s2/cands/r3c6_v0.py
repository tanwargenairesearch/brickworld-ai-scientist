import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 1, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

STEM = 64
WIDTH = 40
HP = 32
HK = 3
MEMB = 8
LAYOUTS = [("TYX", ()), ("TXY", ()), ("TYX", (2,)), ("TYX", (3,))]
D4 = False
REMAP = False
LNORM = False
SEP_CRITIC = False
DILATE = [1, 1]
GBCAST = False
LAYER_ENC = True
LE_IN = 18
LE_HID = 32
LE_OUT = 16
CR_W = 32


def to_grid(v, order, flips, T, Y, X):
    B = v.shape[0]
    if order == "TYX":
        g = v.reshape(B, T, Y, X)
    else:
        g = v.reshape(B, T, X, Y).transpose(2, 3)
    if len(flips) > 0:
        g = g.flip(list(flips))
    return g


def from_grid(m, order, flips):
    if len(flips) > 0:
        m = m.flip(list(flips))
    if order == "TYX":
        out = m
    else:
        out = m.transpose(2, 3)
    return out.reshape(m.shape[0], -1)


def coords(obs, Y, X):
    B = obs.shape[0]
    ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
    xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
    return torch.cat([ys, xs], dim=1)


def sym_views(t, square):
    vs = [t, t.flip([2]), t.flip([3]), t.flip([2, 3])]
    if square:
        tt = t.transpose(2, 3)
        vs = vs + [tt, tt.flip([2]), tt.flip([3]), tt.flip([2, 3])]
    return vs


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.Z = int(spec.Z)
        self.P = self.Y * self.X
        self.C = int(spec.in_channels)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.grid_ok = (self.n_place == self.T * self.P) and (self.n_actions == self.n_place + 1)
        self.square = self.Y == self.X
        self.nviews = 1
        if D4:
            self.nviews = 8 if self.square else 4
        self.layer_ok = LAYER_ENC and (self.C == 4 * self.Z + 3)
        cin = self.C * self.nviews + 2
        if REMAP:
            self.obs_remap = nn.Linear(self.P, self.P)
            cin += self.C
        if self.grid_ok:
            cin += len(LAYOUTS) * self.T
            if REMAP:
                self.mask_remap = nn.Linear(self.P, self.P)
                cin += self.T
        else:
            self.mask_embed = nn.Linear(self.n_actions, MEMB * self.P)
            cin += MEMB
        if self.layer_ok:
            self.le1 = nn.Conv2d(LE_IN, LE_HID, 1)
            self.le2 = nn.Conv2d(LE_HID, LE_OUT, 3, padding=1)
            cin += 2 * LE_OUT
        self.stem1 = nn.Conv2d(cin, STEM, 1)
        if LNORM:
            self.ln = nn.LayerNorm(STEM)
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=d, dilation=d) for d in DILATE])
        if GBCAST:
            self.gb = nn.Linear(2 * WIDTH, WIDTH)
        self.hconv = nn.Conv2d(WIDTH, HP, HK, padding=HK // 2)
        self.hskip = nn.Conv2d(WIDTH, HP, 1)
        if self.grid_ok:
            self.outs = nn.ModuleList([nn.Conv2d(HP, self.T, 1) for _ in LAYOUTS])
            if REMAP:
                self.out_r = nn.Conv2d(HP, self.T, 1)
                self.out_remap = nn.Linear(self.P, self.P)
        else:
            self.out_proj = nn.Conv2d(HP, 4, 1)
            self.out_lin = nn.Linear(4 * self.P, self.n_actions - 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        if SEP_CRITIC:
            self.cr1 = nn.Conv2d(cin, CR_W, 1)
            self.cr2 = nn.Conv2d(CR_W, CR_W, 3, padding=1)
            self.crg = nn.Sequential(nn.Linear(2 * CR_W, 64), nn.ReLU())
        self.value_head = nn.Linear(64, 1)

    def layer_feats(self, obs):
        B = obs.shape[0]
        Y, X, Z = self.Y, self.X, self.Z
        g = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        dn = F.pad(g, (0, 0, 0, 0, 1, 0))[:, :, :Z]
        up = F.pad(g, (0, 0, 0, 0, 0, 1))[:, :, 1:]
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zp = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        cc = coords(obs, Y, X).unsqueeze(2).expand(B, 2, Z, Y, X)
        u = torch.cat([g, dn, up, gl, zp, cc], dim=1)
        u = u.permute(0, 2, 1, 3, 4).reshape(B * Z, LE_IN, Y, X)
        v = F.relu(self.le2(F.relu(self.le1(u))))
        v = v.view(B, Z, LE_OUT, Y, X)
        return torch.cat([v.amax(1), v.mean(1)], dim=1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, P, T = self.Y, self.X, self.P, self.T
        m = mask.float()
        if self.nviews > 1:
            parts = sym_views(obs, self.square)
        else:
            parts = [obs]
        parts.append(coords(obs, Y, X))
        if REMAP:
            parts.append(self.obs_remap(obs.reshape(B, self.C, P)).view(B, self.C, Y, X))
        if self.grid_ok:
            mp = m[:, :self.n_place]
            for o, fl in LAYOUTS:
                parts.append(to_grid(mp, o, fl, T, Y, X))
            if REMAP:
                parts.append(self.mask_remap(mp.reshape(B, T, P)).view(B, T, Y, X))
        else:
            parts.append(self.mask_embed(m).view(B, MEMB, Y, X))
        if self.layer_ok:
            parts.append(self.layer_feats(obs))
        inp = torch.cat(parts, dim=1)
        s = self.stem1(inp)
        if LNORM:
            s = self.ln(s.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = F.relu(self.stem2(F.relu(s)))
        for i, conv in enumerate(self.res):
            x = F.relu(x + conv(x))
            if GBCAST and i == 0:
                x = x + self.gb(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))[:, :, None, None]
        f = x
        h = F.relu(self.hconv(f) + self.hskip(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        if self.grid_ok:
            place = 0
            for k, (o, fl) in enumerate(LAYOUTS):
                place = place + from_grid(self.outs[k](h), o, fl)
            if REMAP:
                place = place + self.out_remap(self.out_r(h).reshape(B, T, P)).reshape(B, -1)
        else:
            place = self.out_lin(self.out_proj(h).flatten(1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        if SEP_CRITIC:
            c = F.relu(self.cr1(inp))
            c = F.relu(c + self.cr2(c))
            v = self.value_head(self.crg(torch.cat([c.mean((2, 3)), c.amax((2, 3))], dim=1)))
        else:
            v = self.value_head(g)
        return logits, v.squeeze(1)


def build_policy(spec):
    return Net(spec)
