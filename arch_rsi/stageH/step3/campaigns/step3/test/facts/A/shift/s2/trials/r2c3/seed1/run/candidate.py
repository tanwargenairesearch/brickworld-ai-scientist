import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
D = 48
NB = 2
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


class MixBlock(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.tok1 = nn.Linear(n, n)
        self.tok2 = nn.Linear(n, n)
        self.ln2 = nn.LayerNorm(d)
        self.ch1 = nn.Linear(d, 2 * d)
        self.ch2 = nn.Linear(2 * d, d)

    def forward(self, x):
        h = self.ln1(x).transpose(1, 2)
        h = self.tok2(F.gelu(self.tok1(h))).transpose(1, 2)
        x = x + h
        x = x + self.ch2(F.gelu(self.ch1(self.ln2(x))))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        n = self.Y * self.X
        self.nt = int(spec.n_types)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, D, 1))
        self.pos = nn.Parameter(torch.randn(1, n, D) * 0.02)
        self.blocks = nn.ModuleList([MixBlock(n, D) for _ in range(NB)])
        self.lnf = nn.LayerNorm(D)
        self.head = PoolHead(D, spec.footprints, PROJ, HID)
        self.aux = nn.Conv2d(D, self.nt, 1)
        self.remap = nn.ModuleList([nn.Linear(n, n, bias=False) for _ in range(self.nt)])
        for r in self.remap:
            nn.init.zeros_(r.weight)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        b = obs.shape[0]
        s = self.stem(obs)
        x = s.flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.lnf(x)
        f = x.transpose(1, 2).reshape(b, D, self.Y, self.X)
        place = self.head(f)
        a = self.aux(f).flatten(2)
        rem = torch.stack([r(a[:, i]) for i, r in enumerate(self.remap)], dim=1).flatten(1)
        place = place + rem
        g = self.glob(torch.cat([x.mean(1), x.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
