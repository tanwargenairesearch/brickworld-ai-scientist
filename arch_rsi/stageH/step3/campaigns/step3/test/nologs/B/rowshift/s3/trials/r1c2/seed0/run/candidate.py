import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

STEM = 48
W = 32
HID = 32
NBLK = 2
GLOB = 64


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.GroupNorm(4, c)
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(4, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.relu(self.n1(x)))
        h = self.c2(F.relu(self.n2(h)))
        return x + h


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.nt = int(spec.n_types)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        cin = spec.in_channels + self.nt
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1))
        self.blocks = nn.Sequential(*[Block(W) for _ in range(NBLK)])
        self.norm = nn.GroupNorm(4, W)
        self.fp = nn.ModuleList([nn.Conv2d(W, HID, (dy, dx)) for dy, dx in self.fps])
        self.fo = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        for o in self.fo:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, GLOB), nn.ReLU(),
                                  nn.Linear(GLOB, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(GLOB, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.nt * self.Y * self.X].float().reshape(B, self.nt, self.Y, self.X)
        x = torch.cat([obs, m], dim=1)
        s = self.stem(x)
        f = self.blocks(s)
        f = F.relu(self.norm(f))
        outs = []
        for (dy, dx), conv, out in zip(self.fps, self.fp, self.fo):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            outs.append(out(F.relu(conv(q))))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
