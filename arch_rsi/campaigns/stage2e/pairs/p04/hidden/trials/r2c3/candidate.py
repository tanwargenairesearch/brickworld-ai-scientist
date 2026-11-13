import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
N_BLOCKS = 2
PROJ = 32
HID = 32

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "gamma": 0.99, "gae_lambda": 0.95}


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.ctx = nn.Linear(2 * c, c)

    def forward(self, x):
        h = F.relu(self.c1(x))
        h = self.c2(h)
        ctx = self.ctx(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        return F.relu(x + h + ctx[:, :, None, None])


class FootprintHead(nn.Module):
    def __init__(self, c, footprints):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in footprints]
        self.proj = nn.Conv2d(c, PROJ, 1)
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * PROJ, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in self.fps
        ])

    def forward(self, f):
        g = F.relu(self.proj(f))
        outs = []
        for (dy, dx), mlp in zip(self.fps, self.mlps):
            if dy > 1 or dx > 1:
                p = F.pad(g, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(p, (dy, dx), stride=1)
                mn = -F.max_pool2d(-p, (dy, dx), stride=1)
            else:
                mx = g
                mn = g
            outs.append(mlp(torch.cat([g, mx, mn], dim=1)))
        return torch.cat(outs, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(WIDTH) for _ in range(N_BLOCKS)])
        self.place_head = FootprintHead(WIDTH, spec.footprints)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = self.blocks(self.stem(obs))
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
