import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8}

W = 40


class Block(nn.Module):
    def __init__(self, c, dil):
        super().__init__()
        self.a = nn.Conv2d(c, c, 3, padding=dil, dilation=dil)
        self.b = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.b(F.relu(self.a(x))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, 64, 1), nn.ReLU(),
            nn.Conv2d(64, W, 1), nn.ReLU())
        self.blocks = nn.Sequential(Block(W, 1), Block(W, 2), Block(W, 1))
        self.head = nn.Conv2d(W, spec.n_types, 5, padding=2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.gate = nn.Parameter(torch.zeros(8))
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        nn.init.zeros_(self.done_head.weight)
        nn.init.zeros_(self.done_head.bias)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs, mask):
        f = self.blocks(self.stem(obs))
        o = self.head(f)
        w = torch.softmax(self.gate, dim=0)
        place = None
        i = 0
        for k in range(4):
            a = o
            if k % 2 == 1:
                a = torch.flip(a, dims=[2])
            if k >= 2:
                a = torch.flip(a, dims=[3])
            for t in range(2):
                b = a if t == 0 else a.transpose(2, 3)
                term = w[i] * b.reshape(b.shape[0], -1)
                place = term if place is None else place + term
                i += 1
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
