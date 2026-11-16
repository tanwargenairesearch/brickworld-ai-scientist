import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.001, 'n_envs': 256, 'rollout_len': 16, 'n_minibatches': 64, 'ppo_epochs': 1, 'ent_coef': 0.02, 'gamma': 0.9, 'gae_lambda': 0.8, 'clip_eps': 0.2}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
GOBS = 16
GW = 64


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def init_all(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init_layer(m, math.sqrt(2.0))


def pool3(p, dy, dx):
    if dy * dx == 1:
        return p, p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        cin = int(spec.in_channels)
        self.stem1 = nn.Conv2d(cin, STEM, 1)
        self.ns = nn.GroupNorm(4, STEM)
        self.stem2 = nn.Conv2d(STEM, COL, 1)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.pmix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.pout = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.type_bias = nn.Parameter(torch.zeros(T))
        self.gobs = nn.Conv2d(cin, GOBS, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + 2 * GOBS, GW), nn.ReLU())
        self.done_head = nn.Linear(GW, 1)
        self.value_head = nn.Linear(GW, 1)
        init_all(self)
        for o in self.pout:
            init_layer(o, 0.01)
        init_layer(self.done_head, 0.01)
        init_layer(self.value_head, 1.0)

    def forward(self, obs, mask):
        s = F.relu(self.stem2(F.relu(self.ns(self.stem1(obs)))))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.pmix, self.pout):
            mx, mn, av = pool3(p, dy, dx)
            res.append(out(F.relu(mix(torch.cat([mx, mn, av], dim=1)))))
        place = torch.cat(res, dim=1) + self.type_bias.view(1, -1, 1, 1)
        go = F.relu(self.gobs(obs))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), go.mean((2, 3)), go.amax((2, 3))], dim=1))
        logits = torch.cat([place.flatten(1), self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
