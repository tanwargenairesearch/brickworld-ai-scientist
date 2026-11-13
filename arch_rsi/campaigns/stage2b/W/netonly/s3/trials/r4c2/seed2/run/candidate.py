import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 32
BLOCKS = 2
HEAD = 16
LH = 20


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.layer_in = nn.Conv2d(13, LH, 1)
        self.glob_in = nn.Conv2d(3, LH, 1)
        self.layer_mix = nn.Conv2d(3 * LH, LH, 1)
        self.stem = nn.Conv2d(2 * LH, WIDTH, 1)
        self.register_buffer("zpos", torch.arange(self.Z).float() / max(self.Z - 1, 1))
        self.blocks = nn.Sequential(*[Block(WIDTH) for _ in range(BLOCKS)])
        self.proj = nn.Conv2d(WIDTH, HEAD, 1)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * HEAD, HEAD, 1), nn.ReLU(), nn.Conv2d(HEAD, 1, 1))
            for _ in self.footprints
        ])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        v = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        pz = F.pad(v, (0, 0, 0, 0, 1, 1))
        below = pz[:, :, :Z]
        above = pz[:, :, 2:]
        zc = self.zpos.to(obs.dtype).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        x = torch.cat([v, below, above, zc], dim=1)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * Z, 13, Y, X)
        gl = self.glob_in(obs[:, 4 * Z:4 * Z + 3]).unsqueeze(1)
        h = F.relu(self.layer_in(x).reshape(B, Z, LH, Y, X) + gl)
        hp = F.pad(h, (0, 0, 0, 0, 0, 0, 1, 1))
        hb = hp[:, :Z]
        ha = hp[:, 2:]
        m = torch.cat([hb, h, ha], dim=2).reshape(B * Z, 3 * LH, Y, X)
        h = h + F.relu(self.layer_mix(m)).reshape(B, Z, LH, Y, X)
        s = torch.cat([h.amax(1), h.mean(1)], dim=1)
        f = F.relu(self.blocks(self.stem(s)))
        p = self.proj(f)
        outs = []
        for (dy, dx), head in zip(self.footprints, self.heads):
            pp = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(pp, (dy, dx), stride=1)
            av = F.avg_pool2d(pp, (dy, dx), stride=1)
            outs.append(head(torch.cat([p, mx, av], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
