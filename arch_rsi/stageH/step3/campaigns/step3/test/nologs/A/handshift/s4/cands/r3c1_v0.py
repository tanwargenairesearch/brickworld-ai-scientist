import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.97, "gae_lambda": 0.95, "lr": 0.001, "ent_coef": 0.005, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32}

W = 48
NBLK = 3
GH = 64


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.ctx = nn.Linear(c, c)

    def forward(self, h):
        g = self.ctx(h.mean((2, 3)))
        return F.relu(h + self.conv(h) + g[:, :, None, None])


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = int(spec.Y) * int(spec.X)
        na = int(spec.n_actions)
        fc = max(2, min(8, 150000 // max(1, n * na)))
        self.stem = nn.Conv2d(spec.in_channels, W, 1)
        self.blocks = nn.ModuleList([Block(W) for _ in range(NBLK)])
        self.red = nn.Conv2d(W, fc, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU())
        self.pi = nn.Linear(fc * n + GH, na)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        h = F.relu(self.stem(obs))
        for b in self.blocks:
            h = b(h)
        z = F.relu(self.red(h)).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = self.pi(torch.cat([z, g], dim=1))
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
