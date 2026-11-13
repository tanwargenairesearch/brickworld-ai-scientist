import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 32
BLOCKS = 2


def cln(ln, x):
    return ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.c1 = nn.Conv2d(w, w, 3, padding=1)
        self.n1 = nn.LayerNorm(w)
        self.c2 = nn.Conv2d(w, w, 3, padding=1)
        self.n2 = nn.LayerNorm(w)

    def forward(self, x):
        h = F.relu(cln(self.n1, self.c1(x)))
        h = cln(self.n2, self.c2(h))
        return F.relu(x + h)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.stem_ln = nn.LayerNorm(WIDTH)
        self.blocks = nn.Sequential(*[Block(WIDTH) for _ in range(BLOCKS)])
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.heads = nn.ModuleList([nn.Conv2d(WIDTH, 1, (dy, dx)) for dy, dx in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.LayerNorm(64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = F.relu(cln(self.stem_ln, self.stem(obs)))
        f = self.blocks(f)
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            p = F.pad(f, (0, dx - 1, 0, dy - 1))
            outs.append(head(p).flatten(1))
        place = torch.cat(outs, dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
