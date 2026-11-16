import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.01, "gamma": 0.97, "gae_lambda": 0.95}
READINGS = ["cyx", "yxc", "ycx"]
OBS_SYM = True
MASK_SYM = True
MASK_MINOR = True
SEP_CRITIC = True
VGRAD = 0.05
STEM = 64
WIDTH = 40
HEAD = 32
GLOB = 64
MASK_EMB = 8
CRIT = 32


def sym(t, tr, fy, fx):
    if tr:
        t = t.transpose(-1, -2)
    if fy:
        t = t.flip(-2)
    if fx:
        t = t.flip(-1)
    return t


class ChanLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def pooled(x):
    return torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.ctx = nn.Linear(2 * c, c)
        self.norm = ChanLN(c)

    def forward(self, x):
        y = x + self.conv(x) + self.ctx(pooled(x))[:, :, None, None]
        return F.relu(self.norm(y))


class Critic(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.c1 = nn.Conv2d(cin, CRIT, 1)
        self.n1 = ChanLN(CRIT)
        self.c2 = nn.Conv2d(CRIT, CRIT, 3, stride=2, padding=1)
        self.fc1 = nn.Linear(2 * CRIT, 64)
        self.fc2 = nn.Linear(64, 1)
        self.log_scale = nn.Parameter(torch.zeros(1))
        nn.init.orthogonal_(self.fc2.weight, gain=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, o):
        h = F.relu(self.n1(self.c1(o)))
        h = F.relu(self.c2(h))
        h = F.relu(self.fc1(pooled(h)))
        return self.fc2(h).squeeze(1) * torch.exp(self.log_scale)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        cells = self.Y * self.X
        square = self.Y == self.X
        if OBS_SYM:
            syms = [(t, a, b) for t in (0, 1) for a in (0, 1) for b in (0, 1)]
            self.syms = [s for s in syms if square or not s[0]]
        else:
            self.syms = [(0, 0, 0)]
        view_ch = self.C * len(READINGS) * len(self.syms)
        self.grid = (self.n_place % cells == 0)
        if self.grid:
            self.K = self.n_place // cells
            n_mb = 2 if MASK_MINOR else 1
            n_ms = len(self.syms) if MASK_SYM else 1
            mask_in = self.K * n_mb * n_ms
        else:
            self.K = 1
            self.mask_embed = nn.Linear(self.n_place, cells * MASK_EMB)
            mask_in = MASK_EMB
        self.obs_stem = nn.Sequential(nn.Conv2d(view_ch + 2, STEM, 1), ChanLN(STEM), nn.ReLU())
        self.mask_stem = nn.Sequential(nn.Conv2d(mask_in, STEM, 1), ChanLN(STEM), nn.ReLU())
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.n0 = ChanLN(WIDTH)
        self.layers = nn.ModuleList([ResBlock(WIDTH), ResBlock(WIDTH)])
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        if self.grid:
            self.out = nn.Conv2d(HEAD, self.K, 1)
        else:
            self.out = nn.Conv2d(HEAD, MASK_EMB, 1)
            self.place_lin = nn.Linear(MASK_EMB * cells, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        outs = [self.out, self.done_head]
        if SEP_CRITIC:
            self.critic = Critic(self.C + 2)
        else:
            self.value_head = nn.Linear(GLOB, 1)
            outs.append(self.value_head)
        for m in outs:
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.zeros_(m.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K, C = self.Y, self.X, self.K, self.C
        bases = []
        for r in READINGS:
            if r == "yxc":
                bases.append(obs.reshape(B, Y, X, C).permute(0, 3, 1, 2))
            elif r == "ycx":
                bases.append(obs.reshape(B, Y, C, X).permute(0, 2, 1, 3))
            else:
                bases.append(obs)
        views = torch.cat([sym(o, *s) for o in bases for s in self.syms], dim=1)
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        mp = mask[:, :self.n_place].to(obs.dtype)
        if self.grid:
            mbases = [mp.reshape(B, K, Y, X)]
            if MASK_MINOR:
                mbases.append(mp.reshape(B, Y, X, K).permute(0, 3, 1, 2))
            msyms = self.syms if MASK_SYM else [(0, 0, 0)]
            mi = torch.cat([sym(m, *s) for m in mbases for s in msyms], dim=1)
        else:
            mi = self.mask_embed(mp).reshape(B, MASK_EMB, Y, X)
        s = self.obs_stem(torch.cat([views, ys, xs], dim=1)) + self.mask_stem(mi)
        x = F.relu(self.n0(self.stem2(s)))
        for layer in self.layers:
            x = layer(x)
        h = F.relu(self.h3(x) + self.h1(x))
        if self.grid:
            place = self.out(h).reshape(B, -1)
        else:
            place = self.place_lin(self.out(h).flatten(1))
        g = self.glob(pooled(x))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        if SEP_CRITIC:
            v = self.critic(torch.cat([obs, ys, xs], dim=1))
            value = VGRAD * v + (1.0 - VGRAD) * v.detach()
        else:
            value = self.value_head(g).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
