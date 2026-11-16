import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

STEM = 64
W = 48
P = 24
H = 24
G = 64
FH = 192


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n = nn.GroupNorm(4, c)
        self.c = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c(F.relu(self.n(x)))


class FootHead(nn.Module):
    def __init__(self, c, footprints):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, P, 1)
        self.fp = nn.ModuleList([nn.Conv2d(P, H, (dy, dx)) for dy, dx in self.fps])
        self.mix = nn.ModuleList([nn.Conv2d(H + 2 * P, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), fp, mix, out in zip(self.fps, self.fp, self.mix, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            a = F.relu(fp(q))
            if dy * dx == 1:
                mx, mn = p, p
            else:
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([a, mx, mn], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cells = spec.Y * spec.X
        cpc = max(4, min(12, 12288 // cells))
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1))
        self.b1 = Block(W)
        self.b2 = Block(W)
        self.norm = nn.GroupNorm(4, W)
        self.head = FootHead(W, spec.footprints)
        self.red = nn.Conv2d(W, cpc, 1)
        self.ffc = nn.Linear(cpc * cells, FH)
        self.fln = nn.LayerNorm(FH)
        self.fout = nn.Linear(FH, spec.n_place)
        nn.init.normal_(self.fout.weight, std=0.01)
        nn.init.zeros_(self.fout.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)

    def forward(self, obs, mask):
        x = self.stem(obs)
        x = self.b2(self.b1(x))
        f = F.relu(self.norm(x))
        place = self.head(f)
        r = F.relu(self.red(f)).flatten(1)
        place = place + self.fout(F.relu(self.fln(self.ffc(r))))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
