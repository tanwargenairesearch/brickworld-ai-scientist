import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.001, 'n_envs': 256, 'rollout_len': 32, 'n_minibatches': 16, 'ppo_epochs': 2, 'ent_coef': 0.01, 'gamma': 0.95, 'gae_lambda': 0.95}

STEM = 64
W = 40
NB = 3
SKIP = 16
P = 32
PH = 32
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
        Y, X = int(spec.Y), int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        self.register_buffer('coord', make_coords(Y, X))
        cin = int(spec.in_channels) + 2
        self.stem1 = nn.Conv2d(cin, STEM, 1)
        self.ns = nn.GroupNorm(4, STEM)
        self.stem2 = nn.Conv2d(STEM, W, 1)
        self.convs = nn.ModuleList([nn.Conv2d(W, W, 3, padding=1) for _ in range(NB)])
        self.norms = nn.ModuleList([nn.GroupNorm(4, W) for _ in range(NB)])
        self.skip = nn.Conv2d(cin, SKIP, 1)
        hc = W + SKIP
        self.proj = nn.Conv2d(hc, P, 1)
        self.pmix = nn.ModuleList([nn.Conv2d(3 * P, PH, 1) for _ in self.fps])
        self.pout = nn.ModuleList([nn.Conv2d(PH, 1, 1) for _ in self.fps])
        self.type_bias = nn.Parameter(torch.zeros(T))
        self.anchor = nn.Conv2d(hc + T, 1, 3, padding=1)
        self.gobs = nn.Conv2d(cin, GOBS, 1)
        self.glob = nn.Sequential(nn.Linear(2 * hc + 2 * GOBS, GW), nn.ReLU(), nn.Linear(GW, GW), nn.ReLU())
        self.done_head = nn.Linear(GW, 1)
        self.value_head = nn.Linear(GW, 1)
        init_all(self)
        for o in self.pout:
            init_layer(o, 0.01)
        init_layer(self.anchor, 0.01)
        init_layer(self.done_head, 0.01)
        nn.init.constant_(self.done_head.bias, DONE_B)
        init_layer(self.value_head, 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = obs.shape[2], obs.shape[3]
        xin = torch.cat([obs, self.coord.expand(B, -1, -1, -1)], dim=1)
        x = F.relu(self.stem2(F.relu(self.ns(self.stem1(xin)))))
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(x + norm(conv(x)))
        f = torch.cat([x, F.relu(self.skip(xin))], dim=1)
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.pmix, self.pout):
            mx, mn, av = pool3(p, dy, dx)
            res.append(out(F.relu(mix(torch.cat([mx, mn, av], dim=1)))))
        s = torch.cat(res, dim=1) + self.type_bias.view(1, -1, 1, 1)
        a = self.anchor(torch.cat([f, s], dim=1))
        la = F.log_softmax(a.flatten(1), dim=1).view(B, 1, Y, X)
        lt = F.log_softmax(s, dim=1)
        go = F.relu(self.gobs(xin))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), go.mean((2, 3)), go.amax((2, 3))], dim=1))
        d = self.done_head(g).squeeze(1)
        place = (la + lt).flatten(1) + F.logsigmoid(-d).unsqueeze(1)
        logits = torch.cat([place, F.logsigmoid(d).unsqueeze(1)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
