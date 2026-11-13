import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "gamma": 0.98, "gae_lambda": 0.95}

WIDTH = 48
NBLOCKS = 3
PROJ = 32
HID = 32
GW = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(NBLOCKS)])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GW), nn.ReLU())
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.ctx = nn.Linear(GW, PROJ)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        heads = []
        for _ in self.fps:
            h = nn.Sequential(nn.Conv2d(3 * PROJ, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            nn.init.normal_(h[2].weight, std=0.01)
            nn.init.zeros_(h[2].bias)
            heads.append(h)
        self.heads = nn.ModuleList(heads)
        self.done_head = nn.Linear(GW, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(GW, 1)

    def forward(self, obs, mask):
        x = self.stem(obs)
        for conv in self.blocks:
            x = x + conv(F.relu(x))
        f = F.relu(x)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        p = F.relu(self.proj(f) + self.ctx(g)[:, :, None, None])
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            pp = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(pp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-pp, (dy, dx), stride=1)
            outs.append(head(torch.cat([p, mx, mn], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
