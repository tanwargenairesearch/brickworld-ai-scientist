import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "gamma": 0.95, "gae_lambda": 0.95}

STEM = 48
WIDTH = 32
K = 12
HID = 32
GH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = int(spec.in_channels)
        self.square = int(spec.X) == int(spec.Y)
        self.T = int(spec.n_types)
        ns = 8 if self.square else 4
        self.stem = nn.Sequential(nn.Conv2d(C, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, STEM, 1), nn.ReLU())
        self.c1 = nn.Conv2d(STEM, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.proj = nn.Conv2d(WIDTH, K, 1)
        self.mix = nn.Conv2d(ns * K, HID, 3, padding=1)
        self.h2 = nn.Conv2d(HID, HID, 3, padding=1)
        self.out = nn.Conv2d(HID, self.T, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.gb = nn.Linear(GH, HID)
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        p = self.proj(f)
        maps = [p, p.flip(2), p.flip(3), p.flip(2).flip(3)]
        if self.square:
            t = p.transpose(2, 3)
            maps = maps + [t, t.flip(2), t.flip(3), t.flip(2).flip(3)]
        h = F.relu(self.mix(torch.cat(maps, dim=1)) + self.gb(g)[:, :, None, None])
        h = F.relu(h + self.h2(h))
        place = self.out(h).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
