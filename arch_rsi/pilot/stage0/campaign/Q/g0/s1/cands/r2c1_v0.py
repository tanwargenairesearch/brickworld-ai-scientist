import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 40
BLOCKS = 2
HEAD = 24


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.c1 = nn.Conv2d(w, w, 3, padding=1)
        self.c2 = nn.Conv2d(w, w, 3, padding=1)
        self.g = nn.Linear(w, w)

    def forward(self, x):
        g = self.g(x.mean((2, 3)))[:, :, None, None]
        h = F.relu(self.c1(x) + g)
        return F.relu(x + self.c2(h))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([Block(WIDTH) for _ in range(BLOCKS)])
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.fp_convs = nn.ModuleList([nn.Conv2d(WIDTH, HEAD, (dy, dx)) for dy, dx in self.fps])
        self.fp_out = nn.ModuleList([nn.Conv2d(HEAD, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        for b in self.blocks:
            f = b(f)
        outs = []
        for (dy, dx), conv, out in zip(self.fps, self.fp_convs, self.fp_out):
            h = F.relu(conv(F.pad(f, (0, dx - 1, 0, dy - 1))))
            outs.append(out(h))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
