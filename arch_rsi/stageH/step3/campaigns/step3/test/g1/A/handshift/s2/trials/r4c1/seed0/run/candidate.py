import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.001, 'n_envs': 128, 'rollout_len': 32, 'n_minibatches': 16, 'ppo_epochs': 2, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_eps': 0.3, 'ent_coef': 0.005}

H = 64
W = 48
SK = 16
K = 32
HID = 32
G = 64


def ortho(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


def cln(norm, x):
    return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def mm_pool(s, dy, dx):
    if dy * dx == 1:
        return s, s
    q = F.pad(s, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    return mn, mx


def add_coords(obs):
    B, C, Y, X = obs.shape
    yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
    xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
    return torch.cat([obs, yy, xx], dim=1)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n = nn.GroupNorm(4, c)
        self.c = ortho(nn.Conv2d(c, c, 3, padding=1), 1.0)

    def forward(self, x):
        return x + self.c(F.relu(self.n(x)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = spec.in_channels + 2
        T = len(self.fps)
        self.e1 = ortho(nn.Conv2d(C, H, 1))
        self.n1 = nn.LayerNorm(H)
        self.e2 = ortho(nn.Conv2d(H, H, 1), 1.0)
        self.down = ortho(nn.Conv2d(H, W, 1))
        self.b1 = Block(W)
        self.b2 = Block(W)
        self.n3 = nn.LayerNorm(3 * W)
        self.rc = ortho(nn.Conv2d(3 * W, W, 1), 1.0)
        self.skip = ortho(nn.Conv2d(C, SK, 1))
        D = H + W + SK
        self.sf = ortho(nn.Conv2d(D, K, 1), 1.0)
        self.mix = nn.ModuleList([ortho(nn.Conv2d(2 * K + W, HID, 1)) for t in range(T)])
        self.outs = nn.ModuleList([ortho(nn.Conv2d(HID, 1, 1), 0.01) for t in range(T)])
        self.anchor = ortho(nn.Conv2d(D, 1, 1), 0.01)
        self.type_bias = nn.Parameter(torch.zeros(T))
        P = 2 * K + 2 * D
        self.glob = nn.Sequential(ortho(nn.Linear(P, G)), nn.ReLU(), ortho(nn.Linear(G, G)), nn.ReLU())
        self.done = ortho(nn.Linear(G, 1), 0.01)
        self.vnet = nn.Sequential(ortho(nn.Linear(P, G)), nn.ReLU(), ortho(nn.Linear(G, 1), 1.0))

    def forward(self, obs, mask):
        o = add_coords(obs)
        x = F.relu(self.e1(o))
        x = F.relu(x + self.e2(F.relu(cln(self.n1, x))))
        f = self.down(x)
        f = self.b2(self.b1(f))
        rmax = f.amax(3, keepdim=True).expand_as(f)
        cmax = f.amax(2, keepdim=True).expand_as(f)
        f = F.relu(f + self.rc(F.relu(cln(self.n3, torch.cat([f, rmax, cmax], dim=1)))))
        k = F.relu(self.skip(o))
        h = torch.cat([x, f, k], dim=1)
        s = torch.sigmoid(self.sf(h))
        a = self.anchor(h)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            mn, mx = mm_pool(s, dy, dx)
            fm = mm_pool(f, dy, dx)[1]
            z = F.relu(self.mix[i](torch.cat([mn, mx, fm], dim=1)))
            res.append(self.outs[i](z) + a + self.type_bias[i])
        place = torch.cat(res, dim=1).flatten(1)
        pooled = torch.cat([s.amax((2, 3)), s.mean((2, 3)), h.amax((2, 3)), h.mean((2, 3))], dim=1)
        g = self.glob(pooled)
        logits = torch.cat([place, self.done(g)], dim=1)
        return logits, self.vnet(pooled).squeeze(1)


def build_policy(spec):
    return Net(spec)
