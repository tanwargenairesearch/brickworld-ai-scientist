import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

T = 32
L = 24
P = 16
K = 16
G = 64


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n = nn.GroupNorm(4, c)
        self.c = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c(F.relu(self.n(x)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, 48, 1), nn.ReLU(),
            nn.Conv2d(48, T, 1))
        self.b1 = Block(T)
        self.b2 = Block(T)
        self.norm = nn.GroupNorm(4, T)
        self.lay_in = nn.Conv2d(11, L, 1)
        self.ctx = nn.Conv2d(T, L, 1)
        self.lay2 = nn.Conv2d(L, L, 1)
        self.proj = nn.Conv2d(L, P, 1)
        self.fp = nn.ModuleList([nn.Conv2d(P, K, (dy, dx)) for dy, dx in self.fps])
        self.mix = nn.ModuleList([nn.Conv2d(K + 2 * P, K, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(K, 2, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * T, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)

    def layer_inputs(self, obs):
        Z = self.Z
        B, _, Y, X = obs.shape
        cur = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        ext = obs[:, 3 * Z:4 * Z]
        cb = F.pad(cur, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        mb = F.pad(miss, (0, 0, 0, 0, 1, 0))[:, :Z]
        ta = F.pad(tgt, (0, 0, 0, 0, 0, 1))[:, 1:]
        hc = obs[:, 4 * Z:4 * Z + 1].expand(B, Z, Y, X)
        ht = obs[:, 4 * Z + 1:4 * Z + 2].expand(B, Z, Y, X)
        tm = obs[:, 4 * Z + 2:4 * Z + 3].expand(B, Z, Y, X)
        zp = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, Z, 1, 1).expand(B, Z, Y, X)
        s = torch.stack([cur, tgt, miss, ext, cb, mb, ta, hc, ht, tm, zp], dim=2)
        return s.reshape(B * Z, 11, Y, X)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        x = self.stem(obs)
        x = self.b2(self.b1(x))
        f = F.relu(self.norm(x))
        c = self.ctx(f).unsqueeze(1).expand(B, Z, L, Y, X).reshape(B * Z, L, Y, X)
        h = F.relu(self.lay_in(self.layer_inputs(obs)) + c)
        h = h + F.relu(self.lay2(h))
        p = self.proj(h)
        res = []
        for (dy, dx), fp, mix, out in zip(self.fps, self.fp, self.mix, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            a = F.relu(fp(q))
            if dy * dx == 1:
                mx, mn = p, p
            else:
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            o = out(F.relu(mix(torch.cat([a, mx, mn], dim=1))))
            o = o.view(B, Z, 2, Y, X)
            w = torch.softmax(o[:, :, 1], dim=1)
            res.append((w * o[:, :, 0]).sum(1))
        place = torch.stack(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
