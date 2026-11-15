import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8}

W = 40


class Block(nn.Module):
    def __init__(self, c, dil):
        super().__init__()
        self.a = nn.Conv2d(c, c, 3, padding=dil, dilation=dil)
        self.na = nn.GroupNorm(8, c)
        self.b = nn.Conv2d(c, c, 3, padding=1)
        self.nb = nn.GroupNorm(8, c)

    def forward(self, x):
        return F.relu(x + self.nb(self.b(F.relu(self.na(self.a(x))))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = spec.Y
        self.X = spec.X
        self.nt = spec.n_types
        self.n_place = spec.n_place
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + spec.n_types, 64, 1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, W, 1), nn.GroupNorm(8, W), nn.ReLU())
        self.blocks = nn.Sequential(Block(W, 1), Block(W, 2), Block(W, 1))
        self.head = nn.Conv2d(W, spec.n_types, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.LayerNorm(64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        nn.init.zeros_(self.done_head.weight)
        nn.init.zeros_(self.done_head.bias)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.nt, self.Y, self.X)
        f = self.blocks(self.stem(torch.cat([obs, m], dim=1)))
        place = self.head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
