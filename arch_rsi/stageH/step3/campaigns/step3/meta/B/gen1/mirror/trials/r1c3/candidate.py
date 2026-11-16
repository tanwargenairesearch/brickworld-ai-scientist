import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

# action-layout hedges: (type_minor, transpose, flip_y, flip_x)
LAYOUTS = [(0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
DONE_BOTH = False
OBS_T = False
HEAD_K = 5
HEAD_C = 24
STEM = 64
WIDTH = 40


def to_grid(m, k, Y, X, lay):
    mi, tr, fy, fx = lay
    B = m.shape[0]
    if mi == 0 and tr == 0:
        g = m.reshape(B, k, Y, X)
    elif mi == 0:
        g = m.reshape(B, k, X, Y).transpose(2, 3)
    elif tr == 0:
        g = m.reshape(B, Y, X, k).permute(0, 3, 1, 2)
    else:
        g = m.reshape(B, X, Y, k).permute(0, 3, 2, 1)
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    return g


def from_grid(g, lay):
    mi, tr, fy, fx = lay
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    if mi == 0 and tr == 1:
        g = g.transpose(2, 3)
    elif mi == 1 and tr == 0:
        g = g.permute(0, 2, 3, 1)
    elif mi == 1:
        g = g.permute(0, 3, 2, 1)
    return g.reshape(g.shape[0], -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.cin = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        yx = self.Y * self.X
        self.grid_ok = self.n_place > 0 and self.n_place % yx == 0
        self.K = self.n_place // yx if self.grid_ok else 1
        self.offs = [0, 1] if (DONE_BOTH and self.n_actions >= self.n_place + 1) else [0]
        self.use_t = OBS_T and self.Y == self.X
        nh = len(LAYOUTS) * len(self.offs)
        if self.grid_ok:
            mask_ch = nh * self.K
            self.mask_lin = None
        else:
            mask_ch = 4
            self.mask_lin = nn.Linear(self.n_actions, 4 * yx)
        obs_ch = self.cin * (2 if self.use_t else 1)
        self.s1 = nn.Conv2d(obs_ch + mask_ch + 2, STEM, 1)
        self.s2 = nn.Conv2d(STEM, WIDTH, 1)
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hc = nn.Conv2d(WIDTH, HEAD_C, HEAD_K, padding=HEAD_K // 2)
        self.hs = nn.Conv2d(WIDTH, HEAD_C, 1)
        if self.grid_ok:
            self.out = nn.Conv2d(HEAD_C, nh * self.K, 1)
            self.out_lin = None
        else:
            self.out = nn.Conv2d(HEAD_C, 4, 1)
            self.out_lin = nn.Linear(4 * yx, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, len(self.offs))
        self.value_head = nn.Linear(64, 1)

    def assemble(self, place, done, off):
        if off == 0:
            full = torch.cat([place, done], 1)
        else:
            full = torch.cat([done, place], 1)
        n = full.shape[1]
        if n < self.n_actions:
            full = F.pad(full, (0, self.n_actions - n))
        elif n > self.n_actions:
            full = full[:, :self.n_actions]
        return full

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        m = mask.to(obs.dtype)
        parts = [obs]
        if self.use_t:
            parts.append(obs.transpose(2, 3))
        if self.grid_ok:
            for off in self.offs:
                mp = m[:, off:off + self.n_place]
                for lay in LAYOUTS:
                    parts.append(to_grid(mp, self.K, Y, X, lay))
        else:
            parts.append(self.mask_lin(m).view(B, 4, Y, X))
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        parts.append(ys)
        parts.append(xs)
        h = torch.cat(parts, 1)
        s = F.relu(self.s2(F.relu(self.s1(h))))
        f = F.relu(s + self.r1(s))
        f = F.relu(f + self.r2(f))
        hd = F.relu(self.hc(f) + self.hs(f))
        o = self.out(hd)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        d = self.done_head(g)
        total = None
        if self.grid_ok:
            nl = len(LAYOUTS)
            for i, off in enumerate(self.offs):
                place = None
                for j, lay in enumerate(LAYOUTS):
                    c0 = (i * nl + j) * self.K
                    p = from_grid(o[:, c0:c0 + self.K], lay)
                    place = p if place is None else place + p
                full = self.assemble(place, d[:, i:i + 1], off)
                total = full if total is None else total + full
        else:
            place = self.out_lin(o.flatten(1))
            total = self.assemble(place, d[:, :1], 0)
        return total, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
