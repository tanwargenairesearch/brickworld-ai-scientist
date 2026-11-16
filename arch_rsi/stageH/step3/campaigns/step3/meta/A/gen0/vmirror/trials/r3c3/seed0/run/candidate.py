import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "lr": 0.001, "gamma": 0.98, "gae_lambda": 0.95, "ent_coef": 0.003}

D = 64
HEADS = 4
FF = 128
NLAYERS = 2
PROJ = 32
HID = 32


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x):
        B, N, Dm = x.shape
        t = self.qkv(self.n1(x)).reshape(B, N, 3, self.heads, Dm // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(t[0], t[1], t[2])
        a = a.transpose(1, 2).reshape(B, N, Dm)
        x = x + self.o(a)
        return x + self.ff(self.n2(x))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        self.n_place = int(spec.n_place)
        n = int(spec.Y) * int(spec.X)
        self.stem = nn.Sequential(
            nn.Linear(spec.in_channels + self.nt, D), nn.ReLU(),
            nn.Linear(D, D))
        self.pos = nn.Parameter(torch.zeros(1, n, D))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(D, HEADS, FF) for _ in range(NLAYERS)])
        self.norm = nn.LayerNorm(D)
        self.head = PoolHead(D, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.nt, Y, X)
        inp = torch.cat([obs, m], dim=1).flatten(2).transpose(1, 2)
        h = self.stem(inp) + self.pos
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        f = h.transpose(1, 2).reshape(B, D, Y, X)
        place = self.head(f)
        g = self.glob(torch.cat([h.mean(1), h.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
