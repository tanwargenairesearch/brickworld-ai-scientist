import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 32, "n_minibatches": 16, "lr": 0.0007, "ent_coef": 0.003, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
WIDTH = 48
HID = 32
GLOB = 64


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 1)
        self.ctx = nn.Linear(2 * c, c)

    def forward(self, x):
        g = self.ctx(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        h = F.relu(self.conv1(x) + g[:, :, None, None])
        return F.relu(x + self.conv2(h))


class FootHead(nn.Module):
    def __init__(self, c, footprints, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.convs = nn.ModuleList([nn.Conv2d(c, hid, (dy, dx)) for dy, dx in self.fps])
        self.mids = nn.ModuleList([nn.Conv2d(hid, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.orthogonal_(o.weight, gain=0.01)
            nn.init.zeros_(o.bias)

    def forward(self, f):
        res = []
        for (dy, dx), cv, md, out in zip(self.fps, self.convs, self.mids, self.outs):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            h = F.relu(cv(q))
            h = F.relu(md(h))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.blocks = nn.ModuleList([Block(WIDTH) for _ in range(2)])
        self.head = FootHead(WIDTH, spec.footprints, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        nn.init.orthogonal_(self.done_head.weight, gain=0.01)
        nn.init.zeros_(self.done_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs, mask):
        f = self.stem(obs)
        for b in self.blocks:
            f = b(f)
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
