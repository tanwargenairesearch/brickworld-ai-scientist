import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2}

STEM = 64
COL = 48
WIDTH = 40
VW = 32
LAYOUTS = [0, 1, 2, 3, 4, 5, 6, 7]


def lay_in(mp, T, Y, X, k):
    B = mp.shape[0]
    if k >= 4:
        m = mp.reshape(B, T, X, Y).transpose(2, 3)
    else:
        m = mp.reshape(B, T, Y, X)
    if k & 1:
        m = m.flip(2)
    if k & 2:
        m = m.flip(3)
    return m


def lay_out(o, k):
    B = o.shape[0]
    if k & 1:
        o = o.flip(2)
    if k & 2:
        o = o.flip(3)
    if k >= 4:
        o = o.transpose(2, 3)
    return o.reshape(B, -1)


class Net(nn.Module):
    def __init__(self, spec, layouts):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.layouts = list(layouts)
        L = len(self.layouts)
        cin = spec.in_channels + L * self.T + 2
        self.stem1 = nn.Conv2d(cin, STEM, 1)
        self.n1 = nn.GroupNorm(8, STEM)
        self.stem2 = nn.Conv2d(STEM, COL, 1)
        self.n2 = nn.GroupNorm(8, COL)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n3 = nn.GroupNorm(8, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n4 = nn.GroupNorm(8, WIDTH)
        self.out = nn.Conv2d(WIDTH, L * self.T, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.v1 = nn.Conv2d(cin, VW, 1)
        self.vn1 = nn.GroupNorm(8, VW)
        self.v2 = nn.Conv2d(VW, VW, 3, padding=1)
        self.vn2 = nn.GroupNorm(8, VW)
        self.vfc = nn.Sequential(nn.Linear(2 * VW, 64), nn.ReLU())
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T = self.Y, self.X, self.T
        mp = mask[:, :self.n_place].to(obs.dtype)
        ms = [lay_in(mp, T, Y, X, k) for k in self.layouts]
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        h = torch.cat([obs] + ms + [yy, xx], dim=1)
        s = F.relu(self.n1(self.stem1(h)))
        s = F.relu(self.n2(self.stem2(s)))
        x = F.relu(self.n3(self.c1(s)))
        f = F.relu(x + self.n4(self.c2(x)))
        o = self.out(f)
        place = None
        for i, k in enumerate(self.layouts):
            l = lay_out(o[:, i * T:(i + 1) * T], k)
            place = l if place is None else place + l
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        v = F.relu(self.vn1(self.v1(h)))
        v = F.relu(self.vn2(self.v2(v)))
        vg = self.vfc(torch.cat([v.mean((2, 3)), v.amax((2, 3))], dim=1))
        return logits, self.value_head(vg).squeeze(1)


def build_policy(spec):
    return Net(spec, LAYOUTS)
