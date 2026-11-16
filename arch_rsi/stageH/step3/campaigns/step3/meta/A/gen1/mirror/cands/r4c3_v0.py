import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.001, 'n_envs': 128, 'rollout_len': 16, 'n_minibatches': 8, 'ppo_epochs': 1, 'ent_coef': 0.01, 'gamma': 0.97, 'gae_lambda': 0.95}

STEM = 64
W = 40
NB = 2
SKIP = 16
P = 24
D = 24
NH = 2
GW = 64


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


class ActAttn(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.nh = nh
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)

    def forward(self, h):
        B, N, Dm = h.shape
        dh = Dm // self.nh
        z = self.qkv(self.norm(h)).reshape(B, N, 3, self.nh, dh).permute(2, 0, 3, 1, 4)
        q, k, v = z[0], z[1], z[2]
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh), dim=-1)
        out = torch.matmul(a, v).permute(0, 2, 1, 3).reshape(B, N, Dm)
        return h + self.o(out)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y, X = int(spec.Y), int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        self.T = T
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
        self.pmix = nn.ModuleList([nn.Conv2d(3 * P, D, 1) for _ in self.fps])
        self.temb = nn.Parameter(torch.zeros(T, D))
        self.attn = ActAttn(D, NH)
        self.tout = nn.Linear(D, 1)
        self.glob = nn.Sequential(nn.Linear(2 * hc + 2 * D, GW), nn.ReLU(), nn.Linear(GW, GW), nn.ReLU())
        self.done_head = nn.Linear(GW, 1)
        self.value_head = nn.Linear(GW, 1)
        init_all(self)
        nn.init.normal_(self.temb, std=0.1)
        init_layer(self.attn.qkv, 1.0)
        init_layer(self.attn.o, 0.5)
        init_layer(self.tout, 0.01)
        init_layer(self.done_head, 0.01)
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
        toks = []
        for (dy, dx), mix in zip(self.fps, self.pmix):
            mx, mn, av = pool3(p, dy, dx)
            toks.append(F.relu(mix(torch.cat([mx, mn, av], dim=1))))
        h = torch.stack(toks, dim=1) + self.temb.view(1, self.T, D, 1, 1)
        h = h.permute(0, 1, 3, 4, 2).reshape(B, self.T * Y * X, D)
        h = self.attn(h)
        place = self.tout(F.relu(h)).squeeze(2)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), h.mean(1), h.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
