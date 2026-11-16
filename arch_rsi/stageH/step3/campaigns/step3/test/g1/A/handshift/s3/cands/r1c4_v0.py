import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0007, "n_envs": 128, "rollout_len": 32, "n_minibatches": 32, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
D = 48
HEADS = 4
FF = 96
SKIP = 16
PROJ = 48
HID = 32
G = 128
DSK = 16


def ortho(module, gain):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def fp_pool(p, dy, dx):
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
        c = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        n = spec.Y * spec.X
        self.stem = nn.Sequential(
            nn.Conv2d(c, STEM, 1), nn.GroupNorm(8, STEM), nn.ReLU(),
            nn.Conv2d(STEM, D, 1), nn.GroupNorm(8, D), nn.ReLU())
        self.c1 = nn.Conv2d(D, D, 3, padding=1)
        self.n1 = nn.GroupNorm(8, D)
        self.pos = nn.Parameter(torch.zeros(1, n, D))
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.ao = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.ff1 = nn.Linear(D, FF)
        self.ff2 = nn.Linear(FF, D)
        self.skip = nn.Conv2d(c, SKIP, 1)
        hin = D + D + SKIP
        self.proj = nn.Conv2d(hin, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.anchor = nn.Conv2d(hin, 1, 1)
        self.tbias = nn.Parameter(torch.zeros(len(self.fps)))
        self.dsk = nn.Conv2d(c, DSK, 1)
        self.glob = nn.Sequential(nn.Linear(2 * hin + 2 * DSK, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)
        ortho(self, math.sqrt(2))
        ortho(self.qkv, 1.0)
        ortho(self.ao, 1.0)
        ortho(self.ff2, 1.0)
        nn.init.normal_(self.pos, std=0.02)
        for o in self.outs:
            ortho(o, 0.01)
        ortho(self.anchor, 0.01)
        ortho(self.done_head, 0.01)
        ortho(self.value_head, 1.0)

    def forward(self, obs, mask):
        b = obs.shape[0]
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        yy, xx = x.shape[2], x.shape[3]
        n = yy * xx
        dh = D // HEADS
        t = x.flatten(2).transpose(1, 2) + self.pos
        u = self.ln1(t)
        qkv = self.qkv(u).reshape(b, n, 3, HEADS, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh), dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(b, n, D)
        t = t + self.ao(o)
        t = t + self.ff2(F.relu(self.ff1(self.ln2(t))))
        f = t.transpose(1, 2).reshape(b, D, yy, xx)
        h = torch.cat([f, s, self.skip(obs)], dim=1)
        p = self.proj(h)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            mx, mn, av = fp_pool(p, dy, dx)
            res.append(out(F.relu(mix(torch.cat([mx, mn, av], dim=1)))))
        place = torch.cat(res, dim=1) + self.anchor(h) + self.tbias.view(1, -1, 1, 1)
        place = place.flatten(1)
        d = F.relu(self.dsk(obs))
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3)), d.mean((2, 3)), d.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
