import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.002, 'n_envs': 256, 'rollout_len': 16, 'n_minibatches': 8, 'ppo_epochs': 3, 'ent_coef': 0.005, 'gamma': 0.97, 'gae_lambda': 0.95}

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


class GluHead(nn.Module):
    def __init__(self, c, fps, proj, hid):
        super().__init__()
        self.fps = fps
        self.proj = nn.Conv2d(c, proj, 1)
        self.mixa = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in fps])
        self.mixb = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), ma, mb, out in zip(self.fps, self.mixa, self.mixb, self.outs):
            mx, mn, av = pool3(p, dy, dx)
            z = torch.cat([mx, mn, av], dim=1)
            h = F.relu(ma(z)) * torch.sigmoid(mb(z))
            res.append(out(h))
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y, self.X = int(spec.Y), int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        self.T = T
        self.n_place = int(spec.n_place)
        cin = int(spec.in_channels) + T
        self.stem1 = nn.Conv2d(cin, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, COL, 1)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.head = GluHead(WIDTH, self.fps, PROJ, HID)
        self.type_bias = nn.Parameter(torch.zeros(T))
        self.gobs = nn.Conv2d(cin, GOBS, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + 2 * GOBS, GW), nn.ReLU(), nn.Linear(GW, GW), nn.ReLU())
        self.done_head = nn.Linear(GW, 1)
        self.value_head = nn.Linear(GW, 1)
        init_all(self)
        for o in self.head.outs:
            init_layer(o, 0.01)
        init_layer(self.done_head, 0.01)
        init_layer(self.value_head, 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.n_place].reshape(B, self.T, self.Y, self.X).to(obs.dtype)
        xin = torch.cat([obs, m], dim=1)
        s = F.relu(self.stem2(F.relu(self.stem1(xin))))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        place = self.head(f) + self.type_bias.view(1, -1, 1, 1)
        go = F.relu(self.gobs(xin))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), go.mean((2, 3)), go.amax((2, 3))], dim=1))
        logits = torch.cat([place.flatten(1), self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
