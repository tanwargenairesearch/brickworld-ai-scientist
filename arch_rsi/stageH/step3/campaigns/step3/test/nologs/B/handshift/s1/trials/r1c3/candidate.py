import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

STEM = 64
WIDTH = 40
HID = 32
GH = 64


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(1, c)
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv(F.relu(self.norm(x)))


class FootHead(nn.Module):
    def __init__(self, c, footprints, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.convs = nn.ModuleList([nn.Conv2d(c, hid, (dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)

    def forward(self, f):
        res = []
        for (dy, dx), conv, out in zip(self.fps, self.convs, self.outs):
            q = F.pad(f, (0, dx - 1, 0, dy - 1)) if dy * dx > 1 else f
            res.append(out(F.relu(conv(q))))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1))
        self.blocks = nn.Sequential(Block(WIDTH), Block(WIDTH))
        self.norm = nn.GroupNorm(1, WIDTH)
        self.head = FootHead(WIDTH, spec.footprints, HID)
        self.done = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU(), nn.Linear(GH, 1))
        self.value = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU(), nn.Linear(GH, 1))
        nn.init.normal_(self.done[2].weight, std=0.01)
        nn.init.zeros_(self.done[2].bias)

    def forward(self, obs, mask):
        f = F.relu(self.norm(self.blocks(self.stem(obs))))
        place = self.head(f)
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        logits = torch.cat([place, self.done(g)], dim=1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
