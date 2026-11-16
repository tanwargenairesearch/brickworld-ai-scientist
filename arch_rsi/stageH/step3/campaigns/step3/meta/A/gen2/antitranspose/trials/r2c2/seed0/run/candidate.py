import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
SKIP = 16
PROJ = 16
WHID = 24
GHID = 64
NF = 8
ROLLS = [0, 1, -1]


def ortho_init(module, gain):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def frame(u, k):
    if k >= 4:
        u = u.transpose(2, 3)
    m = k % 4
    if m == 1:
        u = torch.flip(u, [3])
    elif m == 2:
        u = torch.flip(u, [2])
    elif m == 3:
        u = torch.flip(u, [2, 3])
    return u


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = nn.Conv2d(cin, SKIP, 1)
        self.out_ch = WIDTH + SKIP

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        k = F.relu(self.skip(obs))
        return torch.cat([x, k], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        T = int(spec.n_types)
        self.trunk = Trunk(spec.in_channels)
        c = self.trunk.out_ch
        self.glob = nn.Sequential(nn.Linear(2 * c, GHID), nn.ReLU())
        self.done_head = nn.Linear(GHID, 1)
        self.value_head = nn.Linear(GHID, 1)
        self.type_bias = nn.Linear(GHID, T)
        self.dense = nn.Linear(GHID, spec.n_actions)
        self.proj = nn.Conv2d(c, PROJ, 1)
        self.win = nn.ModuleList([nn.Conv2d(PROJ, WHID, 3, padding=1) for _ in range(NF)])
        self.outs = nn.ModuleList([nn.Conv2d(WHID, T, 1) for _ in range(NF)])
        self.sel = nn.Parameter(torch.zeros(NF * len(ROLLS)))
        ortho_init(self, math.sqrt(2))
        ortho_init(self.done_head, 0.01)
        ortho_init(self.type_bias, 0.01)
        ortho_init(self.dense, 0.01)
        ortho_init(self.outs, 0.01)
        ortho_init(self.value_head, 1.0)

    def forward(self, obs, mask):
        f = self.trunk(obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        tb = self.type_bias(g)[:, :, None, None]
        done = self.done_head(g)
        p = self.proj(f)
        outs = []
        for k in range(NF):
            u = frame(p, k)
            l = self.outs[k](F.relu(self.win[k](u))) + tb
            v = torch.cat([l.flatten(1), done], dim=1)
            for s in ROLLS:
                outs.append(torch.roll(v, s, dims=1) if s != 0 else v)
        w = F.softmax(5.0 * self.sel, dim=0)
        logits = (torch.stack(outs, 1) * w[None, :, None]).sum(1) + self.dense(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
