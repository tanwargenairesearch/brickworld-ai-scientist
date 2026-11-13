import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 40
BLOCKS = 2
K = 16
HID = 32

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "ppo_epochs": 2}


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.c1 = nn.Conv2d(w, w, 3, padding=1)
        self.c2 = nn.Conv2d(w, w, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.c2(F.relu(self.c1(x))))


class TypeHead(nn.Module):
    def __init__(self, w, k, hid):
        super().__init__()
        self.hidden = nn.Conv2d(w + 2 * k, hid, 1)
        self.out = nn.Conv2d(hid, 1, 1)

    def forward(self, x):
        return self.out(F.relu(self.hidden(x)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.Sequential(*[Block(WIDTH) for _ in range(BLOCKS)])
        self.proj = nn.Conv2d(WIDTH, K, 1)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.heads = nn.ModuleList([TypeHead(WIDTH, K, HID) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        f = self.blocks(f)
        h = self.proj(f)
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            hp = F.pad(h, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(hp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-hp, (dy, dx), stride=1)
            outs.append(head(torch.cat([f, mx, mn], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
