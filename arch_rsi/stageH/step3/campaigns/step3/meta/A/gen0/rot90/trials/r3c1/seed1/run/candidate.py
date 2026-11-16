import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.01}

STEM = 64
W = 32
NBLOCK = 3
RED = 8


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.g = nn.Linear(2 * c, c)

    def forward(self, x):
        gv = self.g(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        h = F.relu(self.c1(x) + gv[:, :, None, None])
        return F.relu(x + self.c2(h))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        self.npl = int(spec.n_place)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.blocks = nn.ModuleList([Block(W) for _ in range(NBLOCK)])
        self.place = nn.Conv2d(W, self.nt, 3, padding=1)
        self.red = nn.Conv2d(W, RED, 1)
        self.lin = nn.Linear(RED * int(spec.Y) * int(spec.X), self.npl)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = self.stem(obs)
        for b in self.blocks:
            f = b(f)
        pc = self.place(f).flatten(1)
        pl = self.lin(F.relu(self.red(f)).flatten(1))
        place = pc + pl
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
