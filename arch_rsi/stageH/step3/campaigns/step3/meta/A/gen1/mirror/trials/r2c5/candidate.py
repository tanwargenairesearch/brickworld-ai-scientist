import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.001, 'n_envs': 256, 'rollout_len': 16, 'n_minibatches': 16, 'ppo_epochs': 2, 'ent_coef': 0.01, 'gamma': 0.99, 'gae_lambda': 0.95}

SW = 16
LIN = 12
CTX = 16
P = 24
PH = 24
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


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y, X, Z = int(spec.Y), int(spec.X), int(spec.Z)
        self.Z = Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        cin = int(spec.in_channels)
        self.register_buffer('coord', make_coords(Y, X))
        zl = [torch.full((1, 1, Y, X), float(z) / max(Z - 1, 1)) for z in range(Z)]
        self.register_buffer('zpos', torch.cat(zl, dim=0))
        self.lin = nn.Conv2d(4 + 3 + 1 + 2, LIN, 1)
        self.up = nn.Conv2d(SW + LIN, 2 * SW, 3, padding=1)
        self.down = nn.Conv2d(2 * SW, 2 * SW, 1)
        self.ctx = nn.Conv2d(cin, CTX, 1)
        hc = 3 * SW + CTX
        self.proj = nn.Conv2d(hc, P, 1)
        self.pmix = nn.ModuleList([nn.Conv2d(3 * P, PH, 1) for _ in self.fps])
        self.pout = nn.ModuleList([nn.Conv2d(PH, 1, 1) for _ in self.fps])
        self.out3 = nn.Conv2d(hc, T, 3, padding=1)
        self.type_bias = nn.Parameter(torch.zeros(T))
        self.glob = nn.Sequential(nn.Linear(2 * hc, GW), nn.ReLU(), nn.Linear(GW, GW), nn.ReLU())
        self.done_head = nn.Linear(GW, 1)
        self.value_head = nn.Linear(GW, 1)
        init_all(self)
        init_layer(self.up, 1.0)
        init_layer(self.down, 1.0)
        for o in self.pout:
            init_layer(o, 0.01)
        init_layer(self.out3, 0.01)
        init_layer(self.done_head, 0.01)
        init_layer(self.value_head, 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = obs.shape[2], obs.shape[3]
        Z = self.Z
        gin = obs[:, 4 * Z:]
        co = self.coord.expand(B, -1, -1, -1)
        h = obs.new_zeros(B, SW, Y, X)
        hs = []
        for z in range(Z):
            e = torch.cat([obs[:, z:4 * Z:Z], gin, self.zpos[z:z + 1].expand(B, -1, -1, -1), co], dim=1)
            e = F.relu(self.lin(e))
            u = self.up(torch.cat([h, e], dim=1))
            cand, gate = u.chunk(2, dim=1)
            gt = torch.sigmoid(gate)
            h = (1.0 - gt) * h + gt * torch.tanh(cand)
            hs.append(h)
        d = obs.new_zeros(B, SW, Y, X)
        for z in range(Z - 1, -1, -1):
            u = self.down(torch.cat([d, hs[z]], dim=1))
            cand, gate = u.chunk(2, dim=1)
            gt = torch.sigmoid(gate)
            d = (1.0 - gt) * d + gt * torch.tanh(cand)
        hmax = torch.stack(hs, dim=0).amax(0)
        c = F.relu(self.ctx(obs))
        f = torch.cat([h, d, hmax, c], dim=1)
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.pmix, self.pout):
            mx, mn, av = pool3(p, dy, dx)
            res.append(out(F.relu(mix(torch.cat([mx, mn, av], dim=1)))))
        place = self.out3(f) + torch.cat(res, dim=1) + self.type_bias.view(1, -1, 1, 1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place.flatten(1), self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
