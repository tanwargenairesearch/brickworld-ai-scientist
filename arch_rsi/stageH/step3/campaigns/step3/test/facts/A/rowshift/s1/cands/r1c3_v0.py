import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 512, "rollout_len": 8, "n_minibatches": 16, "ppo_epochs": 1, "gae_lambda": 0.8, "ent_coef": 0.003, "clip_eps": 0.25}

STEM = 64
W = 48
PROJ = 48
HID = 48
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.convs = nn.ModuleList([nn.Conv2d(W, W, 3, padding=1) for _ in range(3)])
        self.glob = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU())
        self.proj = nn.Conv2d(W, PROJ, 1)
        self.gproj = nn.Linear(G, PROJ)
        self.mix1 = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.mix2 = nn.ModuleList([nn.Conv2d(HID, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(m.bias)
        for o in self.outs:
            nn.init.orthogonal_(o.weight, gain=0.01)
        nn.init.orthogonal_(self.done_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, obs, mask):
        x = self.stem(obs)
        for c in self.convs:
            x = F.relu(x + c(x))
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        p = self.proj(x) + self.gproj(g)[:, :, None, None]
        res = []
        for (dy, dx), m1, m2, out in zip(self.fps, self.mix1, self.mix2, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(m1(torch.cat([mx, mn, av], dim=1)))
            h = F.relu(m2(h))
            res.append(out(h))
        place = torch.cat(res, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
