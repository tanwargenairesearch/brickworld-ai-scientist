import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
W = 40
P = 24
HID = 16
G = 16


def pool_feats(p, dy, dx):
    if dy * dx == 1:
        return torch.cat([p, p, p], dim=1)
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return torch.cat([mx, mn, av], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = int(spec.in_channels)
        shapes = set()
        for a, b in spec.footprints:
            shapes.add((int(a), int(b)))
            shapes.add((int(b), int(a)))
        self.shapes = sorted(shapes)
        nt = int(spec.n_types)
        self.stem = nn.Sequential(
            nn.Conv2d(C, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.convs = nn.ModuleList([nn.Conv2d(W, W, 3, padding=1) for _ in range(3)])
        self.proj = nn.Conv2d(2 * W, P, 1)
        self.mixes = nn.ModuleList([nn.Conv2d(3 * P, HID, 1) for _ in self.shapes])
        self.head = nn.Conv2d(len(self.shapes) * HID, nt, 3, padding=1)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
        self.oglob = nn.Conv2d(C, G, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W + G, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = s
        for c in self.convs:
            x = F.relu(x + c(x))
        p = self.proj(torch.cat([s, x], dim=1))
        hs = []
        for (dy, dx), mix in zip(self.shapes, self.mixes):
            hs.append(F.relu(mix(pool_feats(p, dy, dx))))
        place = self.head(torch.cat(hs, dim=1)).flatten(1)
        og = self.oglob(obs).amax((2, 3))
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3)), og], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
