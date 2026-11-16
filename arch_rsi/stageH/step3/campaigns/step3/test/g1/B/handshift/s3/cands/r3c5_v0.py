import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

# (type_minor, transposed, y_flip, x_flip)
MASK_LAYOUTS = [(0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
OBS_READS = ["cyx"]
# (transposed, y_flip, x_flip)
OBS_SYMS = [(0, 0, 0)]
HEAD_K = 3
DENSE_IN = False
LAYER_ENC = True
LAYER_CH = 12
STEM = 64
WIDTH = 40
PROJ = 32


def read_layout(m, K, Y, X, lay):
    minor, trans, fy, fx = lay
    B = m.shape[0]
    if trans:
        a, c = X, Y
    else:
        a, c = Y, X
    if minor:
        g = m.reshape(B, a, c, K).permute(0, 3, 1, 2)
    else:
        g = m.reshape(B, K, a, c)
    if trans:
        g = g.transpose(2, 3)
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    return g


def write_layout(o, lay):
    minor, trans, fy, fx = lay
    if fy:
        o = o.flip(2)
    if fx:
        o = o.flip(3)
    if trans:
        o = o.transpose(2, 3)
    if minor:
        o = o.permute(0, 2, 3, 1)
    return o.reshape(o.shape[0], -1)


def obs_read(obs, mode, C, Y, X):
    B = obs.shape[0]
    if mode == "yxc":
        return obs.reshape(B, Y, X, C).permute(0, 3, 1, 2)
    if mode == "xyc":
        return obs.reshape(B, X, Y, C).permute(0, 3, 2, 1)
    return obs


def obs_sym(obs, lay):
    trans, fy, fx = lay
    if trans:
        obs = obs.transpose(2, 3)
    if fy:
        obs = obs.flip(2)
    if fx:
        obs = obs.flip(3)
    return obs


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.Z = int(spec.Z)
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        yx = self.Y * self.X
        self.grid_ok = self.n_place > 0 and self.n_place % yx == 0
        self.K = self.n_place // yx if self.grid_ok else 4
        self.obs_syms = [l for l in OBS_SYMS if self.Y == self.X or l[0] == 0]
        L = len(MASK_LAYOUTS)
        cin = self.C * len(OBS_READS) * len(self.obs_syms) + 2
        if self.grid_ok:
            cin += L * self.K
        else:
            self.mask_embed = nn.Linear(self.n_actions, 4 * yx)
            cin += 4
        self.dense_obs = DENSE_IN and self.C * yx * yx <= 600000
        if self.dense_obs:
            self.dch = max(1, min(4, 600000 // (self.C * yx * yx)))
            self.obs_dense = nn.Linear(self.C * yx, self.dch * yx)
            cin += self.dch
        if DENSE_IN:
            self.mask_dense = nn.Linear(self.n_actions, 4 * yx)
            cin += 4
        self.layer_ok = LAYER_ENC and self.C == 4 * self.Z + 3
        if self.layer_ok:
            self.layer_conv = nn.Conv2d(12 + 3 + 1, LAYER_CH, 3, padding=1)
            cin += LAYER_CH * self.Z
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(),
                                  nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hk = nn.Conv2d(WIDTH, PROJ, HEAD_K, padding=HEAD_K // 2)
        self.h1 = nn.Conv2d(WIDTH, PROJ, 1)
        if self.grid_ok:
            self.outs = nn.ModuleList([nn.Conv2d(PROJ, self.K, 1) for _ in range(L)])
        else:
            self.out_map = nn.Conv2d(PROJ, 4, 1)
            self.out_lin = nn.Linear(4 * yx, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def layer_feats(self, obs):
        B = obs.shape[0]
        Y, X, Z = self.Y, self.X, self.Z
        g = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        pad = torch.zeros(B, 4, 1, Y, X, device=obs.device, dtype=obs.dtype)
        gp = torch.cat([pad, g, pad], 2)
        st = torch.stack([gp[:, :, z:z + 3] for z in range(Z)], 1).reshape(B * Z, 12, Y, X)
        hts = obs[:, 4 * Z:].unsqueeze(1).expand(B, Z, 3, Y, X).reshape(B * Z, 3, Y, X)
        zc = torch.linspace(-1.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, Z, 1, 1, 1)
        zc = zc.expand(B, Z, 1, Y, X).reshape(B * Z, 1, Y, X)
        le = F.relu(self.layer_conv(torch.cat([st, hts, zc], 1)))
        return le.reshape(B, Z * LAYER_CH, Y, X)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        m = mask.to(obs.dtype)
        ins = []
        for mode in OBS_READS:
            o = obs_read(obs, mode, self.C, Y, X)
            for lay in self.obs_syms:
                ins.append(obs_sym(o, lay))
        if self.grid_ok:
            mp = m[:, :self.n_place]
            for lay in MASK_LAYOUTS:
                ins.append(read_layout(mp, self.K, Y, X, lay))
        else:
            ins.append(self.mask_embed(m).reshape(B, 4, Y, X))
        if self.dense_obs:
            ins.append(self.obs_dense(obs.reshape(B, -1)).reshape(B, self.dch, Y, X))
        if DENSE_IN:
            ins.append(self.mask_dense(m).reshape(B, 4, Y, X))
        if self.layer_ok:
            ins.append(self.layer_feats(obs))
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        ins += [ys, xs]
        x = self.stem(torch.cat(ins, 1))
        x = F.relu(x + self.r1(x))
        f = F.relu(x + self.r2(x))
        h = F.relu(self.hk(f) + self.h1(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        if self.grid_ok:
            place = 0
            for lay, out in zip(MASK_LAYOUTS, self.outs):
                place = place + write_layout(out(h), lay)
        else:
            place = self.out_lin(F.relu(self.out_map(h)).flatten(1))
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
