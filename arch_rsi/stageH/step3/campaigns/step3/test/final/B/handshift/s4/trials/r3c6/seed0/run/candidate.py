import torch
import torch.nn as nn
import torch.nn.functional as F
import math

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.9}

LAYOUTS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
TYPE_MINOR = False
OBS_SYM = False
DONE_FIRST = False
GACT = True
SEP_CRITIC = True
STEM = 64
WIDTH = 40
HEADW = 32
GLOB = 64
CRIT = 32


def lay_fwd(t, l):
    tr, fy, fx = l
    if tr:
        t = t.transpose(-1, -2)
    if fy:
        t = t.flip(-2)
    if fx:
        t = t.flip(-1)
    return t


def lay_inv(t, l):
    tr, fy, fx = l
    if fx:
        t = t.flip(-1)
    if fy:
        t = t.flip(-2)
    if tr:
        t = t.transpose(-1, -2)
    return t


def obs_views(obs, square):
    vs = []
    for tr in ([0, 1] if square else [0]):
        for fy in (0, 1):
            for fx in (0, 1):
                vs.append(lay_fwd(obs, (tr, fy, fx)))
    return torch.cat(vs, 1)


class Critic(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.c1 = nn.Conv2d(c_in, CRIT, 1)
        self.c2 = nn.Conv2d(CRIT, CRIT, 3, padding=1)
        self.fc = nn.Sequential(nn.Linear(2 * CRIT, GLOB), nn.ReLU(), nn.Linear(GLOB, 1))

    def forward(self, o):
        h = F.relu(self.c1(o))
        h = F.relu(h + self.c2(h))
        return self.fc(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)).squeeze(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        yx = self.Y * self.X
        self.K = (self.n_place + yx - 1) // yx
        self.padn = self.K * yx - self.n_place
        self.square = self.Y == self.X
        self.layouts = [l for l in LAYOUTS if self.square or l[0] == 0]
        nL = len(self.layouts)
        nv = ((8 if self.square else 4) if OBS_SYM else 1)
        c_obs = int(spec.in_channels) * nv + 2
        c_mask = self.K * nL
        self.stem1 = nn.Conv2d(c_obs + c_mask, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(2)])
        self.h3 = nn.Conv2d(WIDTH, HEADW, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEADW, 1)
        self.out = nn.Conv2d(HEADW, self.K * nL, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        if GACT:
            self.gact = nn.Linear(GLOB, self.n_actions)
        if SEP_CRITIC:
            self.critic = Critic(int(spec.in_channels) + 2)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K = self.Y, self.X, self.K
        yx = Y * X
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        o = obs_views(obs, self.square) if OBS_SYM else obs
        o = torch.cat([o, ys, xs], 1)
        off = 1 if DONE_FIRST else 0
        m = mask[:, off:off + self.n_place].to(obs.dtype)
        if m.shape[1] < K * yx:
            m = F.pad(m, (0, K * yx - m.shape[1]))
        if TYPE_MINOR:
            m = m.reshape(B, Y, X, K).permute(0, 3, 1, 2)
        else:
            m = m.reshape(B, K, Y, X)
        mv = torch.cat([lay_fwd(m, l) for l in self.layouts], 1)
        x = F.relu(self.stem1(torch.cat([o, mv], 1)))
        x = F.relu(self.stem2(x))
        for conv in self.blocks:
            x = F.relu(x + conv(x))
        h = F.relu(self.h3(x) + self.h1(x))
        maps = self.out(h)
        place = 0
        for i, l in enumerate(self.layouts):
            place = place + lay_inv(maps[:, i * K:(i + 1) * K], l)
        if TYPE_MINOR:
            place = place.permute(0, 2, 3, 1)
        place = place.reshape(B, K * yx)[:, :self.n_place]
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))
        d = self.done_head(g)
        if DONE_FIRST:
            logits = torch.cat([d, place], 1)
        else:
            logits = torch.cat([place, d], 1)
        extra = self.n_actions - logits.shape[1]
        if extra > 0:
            logits = F.pad(logits, (0, extra))
        logits = logits[:, :self.n_actions]
        if GACT:
            logits = logits + self.gact(g)
        if SEP_CRITIC:
            value = self.critic(torch.cat([obs, ys, xs], 1))
        else:
            value = self.value_head(g).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
