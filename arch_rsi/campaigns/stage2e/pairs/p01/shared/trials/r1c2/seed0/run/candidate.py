import torch
import torch.nn as nn
import torch.nn.functional as F

W = 32
NB = 4
RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "gamma": 0.97, "lr": 0.001}


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.c1 = nn.Conv2d(w, w, 3, padding=1)
        self.n1 = nn.GroupNorm(4, w)
        self.c2 = nn.Conv2d(w, w, 3, padding=1)
        self.n2 = nn.GroupNorm(4, w)

    def forward(self, x):
        y = F.relu(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return F.relu(x + y)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(spec.in_channels, W, 3, padding=1), nn.GroupNorm(4, W), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(W) for _ in range(NB)])
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.bcast = nn.Linear(64, W)
        self.place_head = nn.Conv2d(W, spec.n_types, 1)
        nn.init.orthogonal_(self.place_head.weight, 0.01)
        nn.init.zeros_(self.place_head.bias)
        self.done_head = nn.Linear(64, 1)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        h = self.blocks(self.stem(obs))
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        f = F.relu(h + self.bcast(g)[:, :, None, None])
        place = self.place_head(f).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
