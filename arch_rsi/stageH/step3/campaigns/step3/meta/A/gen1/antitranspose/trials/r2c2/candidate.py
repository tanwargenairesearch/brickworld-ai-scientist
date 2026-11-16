import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 1e-3, 'n_envs': 128, 'rollout_len': 32, 'n_minibatches': 32, 'ppo_epochs': 1, 'ent_coef': 0.01, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_eps': 0.2}

COLH = 96
D = 48
HEADS = 4
PROJ = 32
HID = 24
G = 4


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, 2 * d)
        self.ff2 = nn.Linear(2 * d, d)

    def forward(self, x):
        b, n, d = x.shape
        hd = d // self.heads
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q = q.reshape(b, n, self.heads, hd).transpose(1, 2)
        k = k.reshape(b, n, self.heads, hd).transpose(1, 2)
        v = v.reshape(b, n, self.heads, hd).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd), dim=-1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(b, n, d)
        x = x + self.o(y)
        x = x + self.ff2(F.relu(self.ff1(self.n2(x))))
        return x


class FootHead(nn.Module):
    def __init__(self, c, gdim, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        n = len(self.fps)
        self.proj = nn.Conv2d(c, proj, 1)
        self.gproj = nn.Linear(gdim, proj)
        self.foot = nn.ModuleList([nn.Conv2d(proj, hid, (dy, dx)) for dy, dx in self.fps])
        self.mix = nn.ModuleList([nn.Conv2d(2 * proj, hid, 1, bias=False) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])
        self.inter = nn.Conv2d(c, n, 1)
        self.tbias = nn.Parameter(torch.zeros(n))

    def forward(self, f, g):
        p = self.proj(f) + self.gproj(g)[:, :, None, None]
        res = []
        for (dy, dx), foot, mix, out in zip(self.fps, self.foot, self.mix, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            if dy * dx == 1:
                mx, mn = p, p
            else:
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            h = F.relu(foot(q) + mix(torch.cat([mx, mn], dim=1)))
            res.append(out(h))
        s = torch.cat(res, dim=1) + self.inter(f) + self.tbias.view(1, -1, 1, 1)
        return s.flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        self.col = nn.Sequential(
            nn.Conv2d(cin, COLH, 1), nn.GroupNorm(G, COLH), nn.ReLU(),
            nn.Conv2d(COLH, D, 1), nn.ReLU())
        self.pos = nn.Parameter(torch.zeros(1, D, spec.Y, spec.X))
        self.block = Block(D, HEADS)
        c = 2 * D
        gdim = 2 * c
        self.glob = nn.Sequential(nn.Linear(gdim, 64), nn.ReLU())
        self.head = FootHead(c, 64, spec.footprints, PROJ, HID)
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Sequential(nn.Linear(gdim, 64), nn.ReLU(), nn.Linear(64, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init_layer(m, math.sqrt(2))
        for m in self.block.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.pos, std=0.02)
        for m in self.head.outs:
            init_layer(m, 0.01)
        init_layer(self.head.inter, 0.01)
        init_layer(self.done_head, 0.01)
        init_layer(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        c0 = self.col(obs)
        t = (c0 + self.pos).flatten(2).transpose(1, 2)
        t = self.block(t)
        a = t.transpose(1, 2).reshape(B, D, Y, X)
        h = torch.cat([c0, a], dim=1)
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        g = self.glob(pooled)
        place = self.head(h, g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(pooled).squeeze(1)


def build_policy(spec):
    return Net(spec)
