import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
SKIP = 24
FEAT = 64
GH = 64
HID = 40


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def init_all(net):
    for mod in net.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            ortho(mod, math.sqrt(2))


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = nn.Conv2d(cin, SKIP, 1)
        self.glob = nn.Sequential(nn.Linear(2 * FEAT, GH), nn.ReLU())

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        f = torch.cat([x, self.skip(obs)], dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        return f, g


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(self.fps)
        self.K = max(max(a, b) for a, b in self.fps)
        self.trunk = Trunk(spec.in_channels)
        self.convs = nn.ModuleList([nn.Conv2d(FEAT, HID, self.K) for _ in range(2)])
        self.outs = nn.ModuleList([nn.Conv2d(HID, self.T, 1) for _ in range(2)])
        self.tmix = nn.Parameter(torch.zeros(self.T, self.T))
        self.wf = nn.Parameter(torch.zeros(2))
        self.dense = nn.Linear(GH, spec.n_actions)
        self.done_head = nn.Linear(GH, 1)
        self.tbias = nn.Linear(GH, self.T)
        self.value_head = nn.Linear(GH, 1)
        init_all(self)
        for o in self.outs:
            ortho(o, 0.01)
        ortho(self.dense, 0.01)
        ortho(self.done_head, 0.01)
        ortho(self.tbias, 0.01)
        ortho(self.value_head, 1.0)

    def forward(self, obs, mask):
        f, g = self.trunk(obs)
        B = f.shape[0]
        a = torch.softmax(5.0 * self.wf, dim=0)
        k = self.K - 1
        place = None
        for i in range(2):
            u = f if i == 0 else f.transpose(2, 3)
            h = F.relu(self.convs[i](F.pad(u, (0, k, 0, k))))
            m = self.outs[i](h).flatten(2)
            m = m + torch.einsum('ts,bsn->btn', self.tmix, m)
            term = a[i] * m
            place = term if place is None else place + term
        place = place + self.tbias(g).unsqueeze(2)
        logits = torch.cat([place.reshape(B, -1), self.done_head(g)], dim=1) + self.dense(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
