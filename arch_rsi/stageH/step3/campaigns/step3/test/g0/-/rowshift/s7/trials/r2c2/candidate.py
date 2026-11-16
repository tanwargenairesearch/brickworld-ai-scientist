import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 64, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "lr": 0.001, "ent_coef": 0.005, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 32
LH = 32
LP = 16
LMIX = 24
P2 = 16
FH = 32


def pool_fp(p, dy, dx):
    if dy * dx == 1:
        return p, p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        Z = self.Z
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.l_in = nn.Conv2d(10 + Z, LH, 1)
        self.l_tr = nn.Conv2d(WIDTH, LH, 1)
        self.l2 = nn.Conv2d(LH, LH, 1)
        self.l_proj = nn.Conv2d(LH, LP, 1)
        self.l_mix = nn.ModuleList([nn.Conv2d(2 * LP, LMIX, 1) for _ in self.fps])
        self.t_proj = nn.Conv2d(WIDTH, P2, 1)
        self.f_mix = nn.ModuleList([nn.Conv2d(LMIX + 3 * P2, FH, 1) for _ in self.fps])
        self.f_out = nn.ModuleList([nn.Conv2d(FH, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        build = obs[:, 0:Z]
        target = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        bdown = F.pad(build, (0, 0, 0, 0, 1, 0))[:, :Z]
        bup = F.pad(build, (0, 0, 0, 0, 0, 1))[:, 1:]
        mdown = F.pad(miss, (0, 0, 0, 0, 1, 0))[:, :Z]
        per = torch.stack([build, target, miss, extra, bdown, bup, mdown], dim=2)
        g3 = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        lin = torch.cat([per, g3, eye], dim=2).reshape(B * Z, 10 + Z, Y, X)
        tr = self.l_tr(f).unsqueeze(1).expand(B, Z, LH, Y, X).reshape(B * Z, LH, Y, X)
        h = F.relu(self.l_in(lin) + tr)
        h = F.relu(h + self.l2(h))
        lp = self.l_proj(h)
        tp = self.t_proj(f)
        res = []
        for (dy, dx), lmix, fmix, fout in zip(self.fps, self.l_mix, self.f_mix, self.f_out):
            mx, mn, _ = pool_fp(lp, dy, dx)
            lh = F.relu(lmix(torch.cat([mx, mn], dim=1)))
            lh = lh.reshape(B, Z, LMIX, Y, X).amax(1)
            tmx, tmn, tav = pool_fp(tp, dy, dx)
            hh = F.relu(fmix(torch.cat([lh, tmx, tmn, tav], dim=1)))
            res.append(fout(hh))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
