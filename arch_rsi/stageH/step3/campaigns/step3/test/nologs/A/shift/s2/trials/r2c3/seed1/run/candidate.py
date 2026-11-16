import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "lr": 0.0015, "gamma": 0.97, "gae_lambda": 0.95, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16}

STEM = 96
WIDTH = 48
HID = 32
GLOB = 64


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
        for (dy, dx), cv, out in zip(self.fps, self.convs, self.outs):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            res.append(out(F.relu(cv(q))))
        return torch.cat(res, dim=1).flatten(1)


class CtxBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.ctx = nn.Linear(2 * c, c)

    def forward(self, x):
        g = self.ctx(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        return F.relu(x + self.conv(x) + g[:, :, None, None])


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.c0 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.block = CtxBlock(WIDTH)
        self.head = FootHead(WIDTH, spec.footprints, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(GLOB, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(s + self.c0(s))
        x = self.block(x)
        place = self.head(x)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
