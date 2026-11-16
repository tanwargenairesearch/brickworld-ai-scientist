import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2.5e-3, "n_envs": 256, "rollout_len": 8, "n_minibatches": 8, "ppo_epochs": 2, "gae_lambda": 0.8, "ent_coef": 0.005}

W = 32
HID = 32
NMAP = 4


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, 48, 1), nn.ReLU(),
            nn.Conv2d(48, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.hconv = nn.ModuleList([nn.Conv2d(W, HID, (dy, dx)) for dy, dx in self.fps])
        self.hout = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.cellmap = nn.Conv2d(W, NMAP, 1)
        self.done = nn.Linear(2 * NMAP, 1)
        self.value = nn.Sequential(nn.Linear(2 * W + 2 * NMAP, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, obs, mask):
        s = self.stem(obs)
        f = F.relu(s + self.c1(s))
        res = []
        for (dy, dx), hc, ho in zip(self.fps, self.hconv, self.hout):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            h = F.relu(hc(q))
            res.append(ho(h))
        place = torch.cat(res, dim=1).flatten(1)
        m = self.cellmap(f)
        md = torch.cat([m.amax((2, 3)), m.mean((2, 3))], dim=1)
        done = self.done(md)
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3)), md], dim=1)
        v = self.value(g).squeeze(1)
        return torch.cat([place, done], dim=1), v


def build_policy(spec):
    return Net(spec)
