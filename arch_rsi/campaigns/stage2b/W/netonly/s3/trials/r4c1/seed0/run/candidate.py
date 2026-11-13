import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 32
BLOCKS = 2
HEAD = 16
LH = 24
LQ = 6


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
        self.layer_mid = nn.Conv2d(LH, LH, 1)
        self.stem = nn.Conv2d(2 * LH, WIDTH, 1)
        self.register_buffer("zpos", torch.arange(self.Z).float() / max(self.Z - 1, 1))
        self.blocks = nn.Sequential(*[Block(WIDTH) for _ in range(BLOCKS)])
        self.proj = nn.Conv2d(WIDTH, HEAD, 1)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * HEAD, HEAD, 1), nn.ReLU(), nn.Conv2d(HEAD, 1, 1))
            for _ in self.footprints
        ])
        self.lproj = nn.Conv2d(LH, LQ, 1)
        self.lheads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * LQ, LQ, 1), nn.ReLU(), nn.Conv2d(LQ, 1, 1))
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
        h = self.layer_in(x).reshape(B, Z, LH, Y, X) + gl
        h = F.relu(h).reshape(B * Z, LH, Y, X)
        h = F.relu(self.layer_mid(h))
        q = self.lproj(h)
        h5 = h.reshape(B, Z, LH, Y, X)
        s = torch.cat([h5.amax(1), h5.mean(1)], dim=1)
        f = F.relu(self.blocks(self.stem(s)))
        p = self.proj(f)
        outs = []
        for (dy, dx), head, lhead in zip(self.footprints, self.heads, self.lheads):
            pp = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(pp, (dy, dx), stride=1)
            av = F.avg_pool2d(pp, (dy, dx), stride=1)
            main = head(torch.cat([p, mx, av], dim=1))
            qq = F.pad(q, (0, dx - 1, 0, dy - 1))
            lm = F.max_pool2d(qq, (dy, dx), stride=1)
            la = F.avg_pool2d(qq, (dy, dx), stride=1)
            ls = lhead(torch.cat([q, lm, la], dim=1)).reshape(B, Z, Y, X).amax(1, keepdim=True)
            outs.append(main + ls)
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
