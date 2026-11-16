import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
PROJ = 16
HID = 24
NL = 4

RECIPE = {"ppo_epochs": 2, "gamma": 0.98, "gae_lambda": 0.95, "ent_coef": 0.003, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32}


def pool(p, dy, dx):
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
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(6 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, NL, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.orthogonal_(o.weight, 0.01)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        p = self.proj(f)
        cache = {}
        outs = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if (dy, dx) not in cache:
                cache[(dy, dx)] = pool(p, dy, dx)
            if (dx, dy) not in cache:
                cache[(dx, dy)] = pool(p, dx, dy)
            h = F.relu(mix(torch.cat([cache[(dy, dx)], cache[(dx, dy)]], dim=1)))
            outs.append(out(h))
        o = torch.stack(outs, dim=1)
        l0 = o[:, :, 0].flatten(1)
        l1 = o[:, :, 1].transpose(2, 3).flatten(1)
        l2 = o[:, :, 2].permute(0, 2, 3, 1).flatten(1)
        l3 = o[:, :, 3].permute(0, 3, 2, 1).flatten(1)
        place = l0 + l1 + l2 + l3
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
