import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 32, "n_minibatches": 32, "ppo_epochs": 4, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}

K = 24
M = 16
NH = 4
OH = 32
T1 = 48
T2 = 24
GH = 128
NIN = 11


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class ChanLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def fp_minmax(q, dy, dx):
    if dy * dx == 1:
        return q, q
    p = F.pad(q, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(p, (dy, dx), stride=1)
    mn = -F.max_pool2d(-p, (dy, dx), stride=1)
    return mx, mn


class TypeHead(nn.Module):
    def __init__(self, fin, tch):
        super().__init__()
        g = math.sqrt(2.0)
        self.mix = init_layer(nn.Linear(fin, M), g)
        self.score = init_layer(nn.Linear(M, NH), 1.0)
        self.h1 = init_layer(nn.Linear((2 * NH + 1) * M, OH), g)
        self.a1 = init_layer(nn.Linear(tch, OH), g)
        self.out = init_layer(nn.Linear(OH, 1), 0.01)

    def forward(self, h, t):
        nz = h.shape[3]
        e = F.relu(self.mix(h))
        w = torch.softmax(self.score(e), dim=3)
        below = (w.flip(3).cumsum(3).flip(3) - w) / nz
        agg = torch.einsum('byxzh,byxzm->byxhm', w, e).flatten(3)
        bel = torch.einsum('byxzh,byxzm->byxhm', below, e).flatten(3)
        z = F.relu(self.h1(torch.cat([agg, bel, e.amax(3)], dim=3)) + self.a1(t))
        return self.out(z).squeeze(3)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z, self.Y, self.X = int(spec.Z), int(spec.Y), int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        g = math.sqrt(2.0)
        zp = (torch.arange(self.Z, dtype=torch.float32) + 0.5) / self.Z
        self.register_buffer("zpos", zp.view(1, 1, self.Z, 1, 1))
        self.l1 = init_layer(nn.Linear(NIN, K), g)
        self.ln1 = nn.LayerNorm(K)
        self.l2 = init_layer(nn.Linear(K, K), g)
        self.t1 = init_layer(nn.Conv2d(spec.in_channels, T1, 1), g)
        self.tn = ChanLN(T1)
        self.t2 = init_layer(nn.Conv2d(T1, T2, 3, padding=1), g)
        self.heads = nn.ModuleList([TypeHead(2 * K, T2) for _ in self.fps])
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * K + 2 * T2, GH), g), nn.ReLU())
        self.done = init_layer(nn.Linear(GH, 1), 0.01)
        self.value = init_layer(nn.Linear(GH, 1), 1.0)

    def layer_input(self, obs):
        B = obs.shape[0]
        Z, Y, X = self.Z, self.Y, self.X
        L = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        below = F.pad(L[:, :2, :Z - 1], (0, 0, 0, 0, 1, 0))
        above = F.pad(L[:, :1, 1:], (0, 0, 0, 0, 0, 1))
        hts = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zp = self.zpos.expand(B, 1, Z, Y, X)
        x = torch.cat([L, below, above, hts, zp], dim=1)
        return x.permute(0, 2, 3, 4, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Z, Y, X = self.Z, self.Y, self.X
        v = self.layer_input(obs)
        c = F.relu(self.ln1(self.l1(v)))
        c = F.relu(self.l2(c))
        cm = c.permute(0, 1, 4, 2, 3).reshape(B, Z * K, Y, X)
        t = F.relu(self.tn(self.t1(obs)))
        t = F.relu(self.t2(t))
        tp = t.permute(0, 2, 3, 1)
        res = []
        for (dy, dx), head in zip(self.fps, self.heads):
            mx, mn = fp_minmax(cm, dy, dx)
            h = torch.cat([mx.reshape(B, Z, K, Y, X), mn.reshape(B, Z, K, Y, X)], dim=2)
            res.append(head(h.permute(0, 3, 4, 1, 2), tp))
        place = torch.stack(res, dim=1).flatten(1)
        gfeat = torch.cat([c.amax((1, 2, 3)), c.mean((1, 2, 3)), t.amax((2, 3)), t.mean((2, 3))], dim=1)
        g = self.glob(gfeat)
        logits = torch.cat([place, self.done(g)], dim=1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
