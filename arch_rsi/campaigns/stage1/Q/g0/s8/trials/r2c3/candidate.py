import math
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
DEPTH = 2
HID = 16
DK = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        layers, c_in = [], spec.in_channels
        for _ in range(DEPTH):
            layers += [nn.Conv2d(c_in, WIDTH, 3, padding=1), nn.ReLU()]
            c_in = WIDTH
        self.body = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(WIDTH)
        self.q = nn.Linear(WIDTH, DK)
        self.k = nn.Linear(WIDTH, DK)
        self.v = nn.Linear(WIDTH, DK)
        self.o = nn.Linear(DK, WIDTH)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.fp_convs = nn.ModuleList([nn.Conv2d(WIDTH, HID, (dy, dx)) for (dy, dx) in self.fps])
        self.fp_outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        B, C, Y, X = f.shape
        t = f.flatten(2).transpose(1, 2)
        n = self.ln(t)
        q = self.q(n)
        k = self.k(n)
        v = self.v(n)
        a = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / math.sqrt(DK), dim=-1)
        t = t + self.o(torch.matmul(a, v))
        f = t.transpose(1, 2).reshape(B, C, Y, X)
        outs = []
        for (dy, dx), conv, out in zip(self.fps, self.fp_convs, self.fp_outs):
            h = conv(F.pad(f, (0, dx - 1, 0, dy - 1)))
            outs.append(out(F.relu(h)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
