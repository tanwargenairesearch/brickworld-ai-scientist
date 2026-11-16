import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0007, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.97, "gae_lambda": 0.92}

STEM = 64
D = 48
FF = 96
HEADS = 4
NL = 2
PROJ = 32
HID = 32
G = 64


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(D)
        self.att = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.n2 = nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, FF), nn.ReLU(), nn.Linear(FF, D))

    def forward(self, x):
        y = self.n1(x)
        x = x + self.att(y, y, y, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.orthogonal_(o.weight, 0.01)
            nn.init.zeros_(o.bias)

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


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.stem = nn.Sequential(
            nn.Conv2d(int(spec.in_channels) + self.T + 2, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, D, 1))
        self.pos = nn.Parameter(torch.zeros(1, Y * X, D))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(NL)])
        self.nf = nn.LayerNorm(D)
        self.head = PoolHead(D, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * D, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(G, 1)
        ys = torch.linspace(-1.0, 1.0, Y).view(1, 1, Y, 1).expand(1, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X).view(1, 1, 1, X).expand(1, 1, Y, X)
        self.register_buffer("coords", torch.cat([ys, xs], dim=1).clone())

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.T, Y, X)
        cc = self.coords.expand(B, 2, Y, X)
        s = self.stem(torch.cat([obs, m, cc], dim=1))
        tok = s.flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.nf(tok)
        f = F.relu(tok.transpose(1, 2).reshape(B, D, Y, X))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
