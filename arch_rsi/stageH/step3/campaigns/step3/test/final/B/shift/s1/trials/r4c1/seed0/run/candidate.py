import torch
import torch.nn as nn
import torch.nn.functional as F

LAYOUTS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
OBS_SYMS = False
OBS_READS = False
USE_LN = False
OUT_GAIN = 0.0
VGRAD = 0.01
STEM = 64
WIDTH = 40
HEAD = 32
GH = 64
RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}


def fwd_layout(t, lay):
    tr, fy, fx = lay
    if tr:
        t = t.transpose(-1, -2)
    if fy:
        t = t.flip(-2)
    if fx:
        t = t.flip(-1)
    return t


def inv_layout(t, lay):
    tr, fy, fx = lay
    if fx:
        t = t.flip(-1)
    if fy:
        t = t.flip(-2)
    if tr:
        t = t.transpose(-1, -2)
    return t


class ChanLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        hw = self.Y * self.X
        self.K = (self.n_place + hw - 1) // hw
        lays = [l for l in LAYOUTS if self.Y == self.X or l[0] == 0]
        if len(lays) == 0:
            lays = [(0, 0, 0)]
        self.lays = lays
        syms = [(0, 0, 0)]
        if OBS_SYMS:
            syms = [(t, a, b) for t in (0, 1) for a in (0, 1) for b in (0, 1) if self.Y == self.X or t == 0]
        self.syms = syms
        n_reads = 3 if OBS_READS else 1
        c_obs = self.C * len(syms) * n_reads + 2
        c_mask = self.K * len(lays)
        if USE_LN:
            self.obs_stem = nn.Sequential(nn.Conv2d(c_obs, STEM, 1), ChanLN(STEM), nn.ReLU())
            self.mask_stem = nn.Sequential(nn.Conv2d(c_mask, STEM, 1), ChanLN(STEM), nn.ReLU())
            self.stem2 = nn.Sequential(nn.Conv2d(STEM, WIDTH, 1), ChanLN(WIDTH), nn.ReLU())
            self.n1 = ChanLN(WIDTH)
            self.n2 = ChanLN(WIDTH)
        else:
            self.stem = nn.Sequential(nn.Conv2d(c_obs + c_mask, STEM, 1), nn.ReLU(),
                                      nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        self.out = nn.Conv2d(HEAD, self.K * len(lays), 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)
        if OUT_GAIN > 0:
            for m in (self.out, self.done_head, self.value_head):
                nn.init.orthogonal_(m.weight, gain=OUT_GAIN)
                nn.init.zeros_(m.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K, C = self.Y, self.X, self.K, self.C
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        views = [obs]
        if OBS_READS:
            flat = obs.reshape(B, -1)
            views.append(flat.reshape(B, Y, X, C).permute(0, 3, 1, 2))
            views.append(flat.reshape(B, X, Y, C).permute(0, 3, 2, 1))
        o = torch.cat([fwd_layout(v, s) for v in views for s in self.syms] + [ys, xs], 1)
        mp = mask[:, :self.n_place].to(obs.dtype)
        pad = K * Y * X - self.n_place
        if pad > 0:
            mp = F.pad(mp, (0, pad))
        mp = mp.reshape(B, K, Y, X)
        m = torch.cat([fwd_layout(mp, l) for l in self.lays], 1)
        if USE_LN:
            x = self.stem2(self.obs_stem(o) + self.mask_stem(m))
            x = F.relu(x + self.n1(self.r1(x)))
            x = F.relu(x + self.n2(self.r2(x)))
        else:
            x = self.stem(torch.cat([o, m], 1))
            x = F.relu(x + self.r1(x))
            x = F.relu(x + self.r2(x))
        h = F.relu(self.h3(x) + self.h1(x))
        maps = self.out(h)
        place = 0
        for i, l in enumerate(self.lays):
            mi = maps[:, i * K:(i + 1) * K]
            place = place + inv_layout(mi, l).reshape(B, -1)
        place = place[:, :self.n_place]
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))
        logits = torch.cat([place, self.done_head(g)], 1)
        u = self.value_head(g).squeeze(1)
        value = u.detach() + VGRAD * (u - u.detach())
        return logits, value


def build_policy(spec):
    return Net(spec)
