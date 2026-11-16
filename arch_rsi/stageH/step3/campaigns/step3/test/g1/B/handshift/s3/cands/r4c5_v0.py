import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 2e-3, 'n_envs': 128, 'rollout_len': 16, 'n_minibatches': 8,
          'ppo_epochs': 2, 'ent_coef': 0.002, 'gamma': 0.9, 'gae_lambda': 0.85}

# (type_minor, transposed, y_flip, x_flip)
SYM_LAYOUTS = [
    (0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (0, 0, 1, 1),
    (0, 1, 0, 0), (0, 1, 1, 0), (0, 1, 0, 1), (0, 1, 1, 1),
    (1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1), (1, 0, 1, 1),
    (1, 1, 0, 0), (1, 1, 1, 0), (1, 1, 0, 1), (1, 1, 1, 1),
]
# valid-anchor-only enumerations: 'pad' (type block t starts at t*Y*X), 'pack' (blocks contiguous)
COMPACT_MODES = ['pad', 'pack']
# 0: placements then DONE; 1: DONE then placements
DONE_GROUPS = [0, 1]
NORM = True
CRITIC_SPLIT = True
VALUE_GRAD = 0.1
HEAD_K = 3
STEM = 64
WIDTH = 40
PROJ = 32
CRIT = 16


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


def compact_index(fps, K, Y, X, mode):
    n = K * Y * X
    g_in = [n] * n
    g_out = [n] * n
    off = 0
    for t in range(K):
        dy, dx = fps[t]
        ny = Y - dy + 1
        nx = X - dx + 1
        if mode == 'pad':
            off = t * Y * X
        if ny <= 0 or nx <= 0:
            continue
        for y in range(ny):
            for x in range(nx):
                a = off + y * nx + x
                if a < n:
                    s = t * Y * X + y * X + x
                    g_in[s] = a
                    g_out[a] = s
        if mode == 'pack':
            off += ny * nx
    return g_in, g_out


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        yx = self.Y * self.X
        self.grid_ok = self.n_place > 0 and self.n_place % yx == 0 and self.n_actions == self.n_place + 1
        self.K = self.n_place // yx if self.grid_ok else 4
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.lays = list(SYM_LAYOUTS)
        ins_l = []
        outs_l = []
        if self.grid_ok and self.K == len(fps):
            for mode in COMPACT_MODES:
                gi, go = compact_index(fps, self.K, self.Y, self.X, mode)
                self.lays.append(len(ins_l))
                ins_l.append(gi)
                outs_l.append(go)
        self.n_comp = len(ins_l)
        if self.n_comp > 0:
            self.register_buffer('comp_in', torch.tensor(ins_l, dtype=torch.long), persistent=False)
            self.register_buffer('comp_out', torch.tensor(outs_l, dtype=torch.long), persistent=False)
        self.groups = list(DONE_GROUPS) if self.grid_ok else [0]
        L = len(self.lays)
        if self.grid_ok:
            mask_ch = L * self.K * len(self.groups)
        else:
            self.mask_embed = nn.Linear(self.n_actions, 4 * yx)
            mask_ch = 4
        if NORM:
            self.stem_o = nn.Conv2d(self.C, STEM, 1)
            self.stem_m = nn.Conv2d(mask_ch + 2, STEM, 1)
            self.gn_o = nn.GroupNorm(8, STEM)
            self.gn_m = nn.GroupNorm(8, STEM)
            self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
            self.gn2 = nn.GroupNorm(8, WIDTH)
        else:
            self.stem = nn.Sequential(nn.Conv2d(self.C + mask_ch + 2, STEM, 1), nn.ReLU(),
                                      nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hk = nn.Conv2d(WIDTH, PROJ, HEAD_K, padding=HEAD_K // 2)
        self.h1 = nn.Conv2d(WIDTH, PROJ, 1)
        if self.grid_ok:
            self.outs = nn.ModuleList([nn.Conv2d(PROJ, self.K, 1) for _ in range(L * len(self.groups))])
        else:
            self.out_map = nn.Conv2d(PROJ, 4, 1)
            self.out_lin = nn.Linear(4 * yx, self.n_actions - 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_heads = nn.ModuleList([nn.Linear(64, 1) for _ in self.groups])
        if CRITIC_SPLIT:
            self.c1 = nn.Conv2d(WIDTH, CRIT, 3, padding=1)
            self.c2 = nn.Conv2d(CRIT, CRIT, 3, padding=1)
            self.vhid = nn.Linear(2 * CRIT + 2 * WIDTH, 64)
            self.value_out = nn.Linear(64, 1)
        else:
            self.value_head = nn.Linear(64, 1)

    def read_any(self, mp, lay):
        if isinstance(lay, int):
            B = mp.shape[0]
            ext = torch.cat([mp, mp.new_zeros(B, 1)], 1)
            return ext[:, self.comp_in[lay]].reshape(B, self.K, self.Y, self.X)
        return read_layout(mp, self.K, self.Y, self.X, lay)

    def write_any(self, o, lay):
        if isinstance(lay, int):
            B = o.shape[0]
            flat = o.reshape(B, -1)
            ext = torch.cat([flat, flat.new_zeros(B, 1)], 1)
            return ext[:, self.comp_out[lay]]
        return write_layout(o, lay)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        m = mask.to(obs.dtype)
        if self.grid_ok:
            feats = []
            for off in self.groups:
                mp = m[:, off:off + self.n_place]
                for lay in self.lays:
                    feats.append(self.read_any(mp, lay))
            mfeat = torch.cat(feats, 1)
        else:
            mfeat = self.mask_embed(m).reshape(B, 4, Y, X)
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        if NORM:
            a = self.gn_o(self.stem_o(obs)) + self.gn_m(self.stem_m(torch.cat([mfeat, ys, xs], 1)))
            x = F.relu(self.gn2(self.stem2(F.relu(a))))
        else:
            x = self.stem(torch.cat([obs, mfeat, ys, xs], 1))
        x0 = x
        x = F.relu(x + self.r1(x))
        f = F.relu(x + self.r2(x))
        h = F.relu(self.hk(f) + self.h1(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        if self.grid_ok:
            logits = 0
            i = 0
            for gi, off in enumerate(self.groups):
                place = 0
                for lay in self.lays:
                    place = place + self.write_any(self.outs[i](h), lay)
                    i += 1
                d = self.done_heads[gi](g)
                if off == 0:
                    lg = torch.cat([place, d], 1)
                else:
                    lg = torch.cat([d, place], 1)
                logits = logits + lg
        else:
            place = self.out_lin(F.relu(self.out_map(h)).flatten(1))
            logits = torch.cat([place, self.done_heads[0](g)], 1)
        if CRITIC_SPLIT:
            c = F.relu(self.c1(x0.detach()))
            c = F.relu(c + self.c2(c))
            fd = f.detach()
            cf = torch.cat([c.mean((2, 3)), c.amax((2, 3)), fd.mean((2, 3)), fd.amax((2, 3))], 1)
            v = self.value_out(F.relu(self.vhid(cf))).squeeze(1)
            value = VALUE_GRAD * v + (1.0 - VALUE_GRAD) * v.detach()
        else:
            value = self.value_head(g).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
