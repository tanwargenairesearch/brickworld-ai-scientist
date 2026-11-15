import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 32, "n_minibatches": 16, "lr": 0.001, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95}

W = 40
HH = 32


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.conv(x))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = len(spec.footprints)
        self.square = int(spec.Y) == int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, 64, 1), nn.ReLU(),
            nn.Conv2d(64, W, 1), nn.ReLU())
        self.blocks = nn.Sequential(Block(W), Block(W), Block(W))
        self.hproj = nn.Conv2d(W, HH, 1)
        self.hconv = nn.Conv2d(HH, HH, 3, padding=1)
        self.hout = nn.Conv2d(HH, 2 * self.T, 1)
        k = 5 if self.square else 4
        self.mix = nn.Conv2d(k * 2 * self.T, self.T, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = self.blocks(self.stem(obs))
        h = F.relu(self.hproj(f))
        h = F.relu(self.hconv(h))
        L = self.hout(h)
        maps = [L, L.flip(2), L.flip(3), L.flip(2).flip(3)]
        if self.square:
            maps.append(L.transpose(2, 3))
        place = self.mix(torch.cat(maps, dim=1)).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
