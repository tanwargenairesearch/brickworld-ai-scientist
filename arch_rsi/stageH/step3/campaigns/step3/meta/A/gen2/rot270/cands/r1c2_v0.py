import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
SKIP = 24
PROJ = 32
HID = 32
GLOB = 64
MIX_SCALE = 5.0


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.s1 = init_layer(nn.Conv2d(cin, STEM, 1))
        self.s2 = init_layer(nn.Conv2d(STEM, COL, 1))
        self.c1 = init_layer(nn.Conv2d(COL, WIDTH, 3, padding=1))
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = init_layer(nn.Conv2d(WIDTH, WIDTH, 3, padding=1))
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = init_layer(nn.Conv2d(cin, SKIP, 1))
        self.out_ch = WIDTH + SKIP

    def forward(self, obs):
        s = F.relu(self.s2(F.relu(self.s1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        k = F.relu(self.skip(obs))
        return f, torch.cat([f, k], 1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.K = max(max(int(a), int(b)) for a, b in spec.footprints)
        self.trunk = Trunk(spec.in_channels)
        c = self.trunk.out_ch
        self.proj = init_layer(nn.Conv2d(c, PROJ, 1))
        self.convs = nn.ModuleList([init_layer(nn.Conv2d(PROJ, HID, self.K)) for _ in range(2)])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID, self.T, 1), 0.01) for _ in range(2)])
        self.tmix = nn.ModuleList([nn.Conv2d(self.T, self.T, 1, bias=False) for _ in range(2)])
        for m in self.tmix:
            nn.init.zeros_(m.weight)
        self.alpha = nn.Parameter(torch.zeros(2))
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * WIDTH, GLOB)), nn.ReLU())
        self.done_head = init_layer(nn.Linear(GLOB, 1), 0.01)
        self.dense = init_layer(nn.Linear(GLOB, spec.n_actions), 0.01)
        self.value_head = init_layer(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        f, h = self.trunk(obs)
        p = F.relu(self.proj(h))
        zs = []
        for i in range(2):
            q = p if i == 0 else p.transpose(2, 3)
            q = F.pad(q, (0, self.K - 1, 0, self.K - 1))
            z = self.outs[i](F.relu(self.convs[i](q)))
            z = z + self.tmix[i](z)
            zs.append(z.flatten(1))
        z = torch.stack(zs, 1)
        w = torch.softmax(MIX_SCALE * self.alpha, 0)
        place = (z * w.view(1, -1, 1)).sum(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        logits = torch.cat([place, self.done_head(g)], 1) + self.dense(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
