import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.98, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0005}

STEM = 64
W = 40
PROJ = 32
GH = 64
NBLOCKS = 3
KOUT = 3
NLAY = 4


def to_grid(m, k, T, Y, X):
    B = m.shape[0]
    if k == 0:
        return m.reshape(B, T, Y, X)
    if k == 1:
        return m.reshape(B, T, X, Y).transpose(2, 3)
    if k == 2:
        return m.reshape(B, Y, X, T).permute(0, 3, 1, 2)
    return m.reshape(B, X, Y, T).permute(0, 3, 2, 1)


def from_grid(s, k):
    if k == 0:
        return s.flatten(1)
    if k == 1:
        return s.transpose(2, 3).flatten(1)
    if k == 2:
        return s.permute(0, 2, 3, 1).flatten(1)
    return s.permute(0, 3, 2, 1).flatten(1)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(1, c)
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv(F.relu(self.norm(x)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        cin = int(spec.in_channels) + NLAY * self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1))
        self.blocks = nn.ModuleList([Block(W) for _ in range(NBLOCKS)])
        self.norm = nn.GroupNorm(1, W)
        self.proj = nn.Conv2d(W, PROJ, 1)
        self.outs = nn.ModuleList([nn.Conv2d(PROJ, self.T, KOUT, padding=KOUT // 2) for _ in range(NLAY)])
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.any_head = nn.Linear(GH, self.n_actions)
        nn.init.normal_(self.any_head.weight, std=0.01)
        nn.init.zeros_(self.any_head.bias)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        T = self.T
        mp = mask[:, :self.n_place].to(obs.dtype)
        ms = [to_grid(mp, k, T, Y, X) for k in range(NLAY)]
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        h = self.stem(torch.cat([obs] + ms + [yy, xx], dim=1))
        for blk in self.blocks:
            h = blk(h)
        f = F.relu(self.norm(h))
        p = F.relu(self.proj(f))
        place = None
        for k, out in enumerate(self.outs):
            s = from_grid(out(p), k)
            place = s if place is None else place + s
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1) + self.any_head(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
