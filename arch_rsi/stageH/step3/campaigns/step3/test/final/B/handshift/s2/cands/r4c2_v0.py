import torch
import torch.nn as nn
import torch.nn.functional as F

LAYOUT = (0, 0, 0)
OBS_SYM = True
TYPE_MINOR = True
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


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        cells = self.Y * self.X
        self.K = (self.n_place + cells - 1) // cells
        sq = self.Y == self.X
        lay = tuple(LAYOUT)
        if not sq:
            lay = (0, lay[1], lay[2])
        self.lay = lay
        if OBS_SYM:
            syms = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1) if sq or a == 0]
        else:
            syms = [(0, 0, 0)]
        self.syms = syms
        c_obs = spec.in_channels * len(syms) + 2
        self.stem1 = nn.Sequential(nn.Conv2d(c_obs + self.K, STEM, 1), nn.ReLU())
        self.stem2 = nn.Sequential(nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(2)])
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        self.hout = nn.Conv2d(HEAD, self.K, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)
        ys = torch.linspace(-1.0, 1.0, self.Y).view(1, 1, self.Y, 1).expand(1, 1, self.Y, self.X)
        xs = torch.linspace(-1.0, 1.0, self.X).view(1, 1, 1, self.X).expand(1, 1, self.Y, self.X)
        self.register_buffer("coords", torch.cat([ys, xs], dim=1).contiguous())

    def forward(self, obs, mask):
        B = obs.shape[0]
        cells = self.Y * self.X
        m = mask[:, :self.n_place].float()
        pad = self.K * cells - self.n_place
        if pad > 0:
            m = F.pad(m, (0, pad))
        if TYPE_MINOR:
            m = m.view(B, self.Y, self.X, self.K).permute(0, 3, 1, 2)
        else:
            m = m.view(B, self.K, self.Y, self.X)
        mfeat = fwd_layout(m, self.lay)
        crd = self.coords.expand(B, -1, -1, -1)
        ofeat = torch.cat([fwd_layout(obs, s) for s in self.syms] + [crd], dim=1)
        x = self.stem2(self.stem1(torch.cat([ofeat, mfeat], dim=1)))
        for conv in self.res:
            x = F.relu(x + conv(x))
        h = F.relu(self.h3(x) + self.h1(x))
        o = inv_layout(self.hout(h), self.lay)
        if TYPE_MINOR:
            o = o.permute(0, 2, 3, 1)
        place = o.reshape(B, -1)[:, :self.n_place]
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
