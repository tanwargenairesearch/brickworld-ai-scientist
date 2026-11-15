import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
HID = 32
GLOB = 64


class WindowHead(nn.Module):
    def __init__(self, c, footprints, hid, gdim):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.win = nn.ModuleList([nn.Conv2d(c, hid, (dy, dx)) for dy, dx in self.fps])
        self.gin = nn.ModuleList([nn.Linear(gdim, hid) for _ in self.fps])
        self.mix = nn.ModuleList([nn.Conv2d(hid, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f, g):
        res = []
        for (dy, dx), win, gin, mix, out in zip(self.fps, self.win, self.gin, self.mix, self.outs):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            h = F.relu(win(q) + gin(g)[:, :, None, None])
            h = F.relu(mix(h))
            res.append(out(h).flatten(1))
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.head = WindowHead(COL + WIDTH, spec.footprints, HID, GLOB)
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Sequential(nn.Linear(GLOB, GLOB), nn.ReLU(), nn.Linear(GLOB, 1))

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        place = self.head(torch.cat([s, f], dim=1), g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
