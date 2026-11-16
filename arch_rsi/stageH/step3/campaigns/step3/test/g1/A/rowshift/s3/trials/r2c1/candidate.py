import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 32, "n_minibatches": 32, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}

D = 64
TN = 64
PROJ = 24
HID = 32
G = 64
RG = 1.4142135623730951


def init_(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def fp_pool(p, dy, dx):
    if dy * dx == 1:
        return [p, p, p]
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return [mx, mn, av]


class MixBlock(nn.Module):
    def __init__(self, n, d, tn):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.t1 = init_(nn.Linear(n, tn), RG)
        self.t2 = init_(nn.Linear(tn, n), 1.0)
        self.dw = nn.Conv2d(d, d, 3, padding=1, groups=d)
        self.ln2 = nn.LayerNorm(d)
        self.c1 = init_(nn.Linear(d, 2 * d), RG)
        self.c2 = init_(nn.Linear(2 * d, d), 1.0)

    def forward(self, x, Y, X):
        B, N, Dd = x.shape
        h = self.ln1(x).transpose(1, 2)
        t = self.t2(F.gelu(self.t1(h))) + self.dw(h.reshape(B, Dd, Y, X)).flatten(2)
        x = x + t.transpose(1, 2)
        x = x + self.c2(F.gelu(self.c1(self.ln2(x))))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        n = self.Y * self.X
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        nt = len(self.fps)
        self.stem = init_(nn.Linear(cin, D), 1.0)
        self.pos = nn.Parameter(torch.zeros(1, n, D))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([MixBlock(n, D, TN) for _ in range(2)])
        self.lnf = nn.LayerNorm(D)
        self.proj = init_(nn.Conv2d(D, PROJ, 1), 1.0)
        self.mix = nn.ModuleList([init_(nn.Conv2d(3 * PROJ, HID, 1), RG) for _ in self.fps])
        self.outs = nn.ModuleList([init_(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_(nn.Conv2d(D, 1, 1), 0.01)
        self.K = max(1, min(4, 200000 // (n * int(spec.n_place))))
        self.dcell = init_(nn.Conv2d(D, self.K, 1), 1.0)
        self.dense = init_(nn.Linear(self.K * n, int(spec.n_place)), 0.01)
        self.gnorm = nn.LayerNorm(2 * D)
        self.glob = init_(nn.Linear(2 * D, G), RG)
        self.type_head = init_(nn.Linear(G, nt), 0.01)
        self.done_head = init_(nn.Linear(G, 1), 0.01)
        self.value_head = init_(nn.Linear(G, 1), 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        tok = obs.flatten(2).transpose(1, 2)
        x = self.stem(tok) + self.pos
        for blk in self.blocks:
            x = blk(x, Y, X)
        x = self.lnf(x)
        g = torch.cat([x.mean(1), x.amax(1)], dim=1)
        g = F.relu(self.glob(self.gnorm(g)))
        f = x.transpose(1, 2).reshape(B, D, Y, X)
        tb = self.type_head(g)
        anc = self.anchor(f)
        p = self.proj(f)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            z = torch.cat(fp_pool(p, dy, dx), dim=1)
            z = F.relu(self.mix[i](z))
            res.append(self.outs[i](z) + anc + tb[:, i].view(-1, 1, 1, 1))
        place = torch.cat(res, dim=1).flatten(1) + self.dense(self.dcell(f).flatten(1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
