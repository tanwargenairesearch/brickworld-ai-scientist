import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.002, 'n_envs': 256, 'rollout_len': 32, 'n_minibatches': 32, 'ppo_epochs': 1, 'ent_coef': 0.02, 'gamma': 0.97, 'gae_lambda': 0.95, 'clip_eps': 0.25}

BS = 48
W = 40
NB = 3
SP = 16
P = 16
HH = 32
GOBS = 16
GW = 64
DONE_B = -3.0


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def init_all(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init_layer(m, math.sqrt(2.0))


def make_coords(Y, X):
    yy = torch.linspace(-1.0, 1.0, Y).view(1, 1, Y, 1).expand(1, 1, Y, X)
    xx = torch.linspace(-1.0, 1.0, X).view(1, 1, 1, X).expand(1, 1, Y, X)
    return torch.cat([yy, xx], dim=1).contiguous()


def stop_gate(place, d):
    lp = F.log_softmax(place, dim=1) + F.logsigmoid(-d).unsqueeze(1)
    return torch.cat([lp, F.logsigmoid(d).unsqueeze(1)], dim=1)


class TokMix(nn.Module):
    def __init__(self, w, n):
        super().__init__()
        self.norm = nn.GroupNorm(4, w)
        self.t1 = nn.Linear(n, n)
        self.t2 = nn.Linear(n, n)

    def forward(self, x):
        B, C, Y, X = x.shape
        h = self.norm(x).reshape(B, C, Y * X)
        h = self.t2(F.relu(self.t1(h)))
        return x + h.reshape(B, C, Y, X)


class ConcatHead(nn.Module):
    def __init__(self, c, fps, p, hh):
        super().__init__()
        self.fps = fps
        self.proj = nn.Conv2d(c, p, 1)
        self.m1 = nn.ModuleList([nn.Conv2d(p * dy * dx, hh, 1) for dy, dx in fps])
        self.m2 = nn.ModuleList([nn.Conv2d(hh, hh, 1) for _ in fps])
        self.outs = nn.ModuleList([nn.Conv2d(hh, 1, 1) for _ in fps])

    def forward(self, f):
        p = self.proj(f)
        Y, X = p.shape[2], p.shape[3]
        res = []
        for (dy, dx), m1, m2, out in zip(self.fps, self.m1, self.m2, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            parts = [q[:, :, i:i + Y, j:j + X] for i in range(dy) for j in range(dx)]
            h = F.relu(m1(torch.cat(parts, dim=1)))
            h = F.relu(m2(h))
            res.append(out(h))
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y, X = int(spec.Y), int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        self.register_buffer('coord', make_coords(Y, X))
        cin = int(spec.in_channels) + 2
        self.sa = nn.Conv2d(cin, BS, 1)
        self.sb = nn.Conv2d(cin, BS, 1)
        self.ns = nn.GroupNorm(4, BS)
        self.stem2 = nn.Conv2d(BS, W, 1)
        self.convs = nn.ModuleList([nn.Conv2d(W, W, 3, padding=1) for _ in range(NB)])
        self.norms = nn.ModuleList([nn.GroupNorm(4, W) for _ in range(NB)])
        self.tok = TokMix(W, Y * X)
        self.sproj = nn.Conv2d(BS, SP, 1)
        hc = W + SP
        self.head = ConcatHead(hc, self.fps, P, HH)
        self.out3 = nn.Conv2d(hc, T, 3, padding=1)
        self.type_bias = nn.Parameter(torch.zeros(T))
        self.gobs = nn.Conv2d(cin, GOBS, 1)
        self.glob = nn.Sequential(nn.Linear(2 * hc + 2 * GOBS, GW), nn.ReLU(), nn.Linear(GW, GW), nn.ReLU())
        self.done_head = nn.Linear(GW, 1)
        self.value_head = nn.Linear(GW, 1)
        init_all(self)
        init_layer(self.sb, 1.0)
        init_layer(self.tok.t2, 0.5)
        for o in self.head.outs:
            init_layer(o, 0.01)
        init_layer(self.out3, 0.01)
        init_layer(self.done_head, 0.01)
        nn.init.constant_(self.done_head.bias, DONE_B)
        init_layer(self.value_head, 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        xin = torch.cat([obs, self.coord.expand(B, -1, -1, -1)], dim=1)
        s = self.ns(F.relu(self.sa(xin)) * self.sb(xin))
        x = F.relu(self.stem2(s))
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(x + norm(conv(x)))
        x = self.tok(x)
        f = torch.cat([x, F.relu(self.sproj(s))], dim=1)
        place = self.head(f) + self.out3(f) + self.type_bias.view(1, -1, 1, 1)
        go = F.relu(self.gobs(xin))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), go.mean((2, 3)), go.amax((2, 3))], dim=1))
        logits = stop_gate(place.flatten(1), self.done_head(g).squeeze(1))
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
