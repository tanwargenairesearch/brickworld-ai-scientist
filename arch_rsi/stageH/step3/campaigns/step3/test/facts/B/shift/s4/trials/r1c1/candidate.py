import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
LAYOUTS = [(False, False, False), (True, False, False), (False, True, False), (False, False, True)]


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.X = int(spec.X)
        self.Y = int(spec.Y)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.L = len(LAYOUTS)
        cin = int(spec.in_channels) + self.L * self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.out = nn.Conv2d(WIDTH, self.L * self.T, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        ys = torch.linspace(-1.0, 1.0, self.Y).view(1, 1, self.Y, 1).expand(1, 1, self.Y, self.X)
        xs = torch.linspace(-1.0, 1.0, self.X).view(1, 1, 1, self.X).expand(1, 1, self.Y, self.X)
        self.register_buffer("coords", torch.cat([ys, xs], dim=1).contiguous())

    def to_grid(self, m, lay):
        tr, fy, fx = lay
        B = m.shape[0]
        if tr:
            g = m.reshape(B, self.T, self.X, self.Y).transpose(2, 3)
        else:
            g = m.reshape(B, self.T, self.Y, self.X)
        if fy:
            g = g.flip(2)
        if fx:
            g = g.flip(3)
        return g

    def from_grid(self, g, lay):
        tr, fy, fx = lay
        B = g.shape[0]
        if fy:
            g = g.flip(2)
        if fx:
            g = g.flip(3)
        if tr:
            g = g.transpose(2, 3)
        return g.reshape(B, -1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.n_place].to(obs.dtype)
        chans = [obs] + [self.to_grid(m, lay) for lay in LAYOUTS] + [self.coords.expand(B, -1, -1, -1)]
        x = torch.cat(chans, dim=1)
        s = self.stem(x)
        h = F.relu(self.c1(s))
        f = F.relu(h + self.c2(h))
        o = self.out(f)
        place = None
        for i, lay in enumerate(LAYOUTS):
            p = self.from_grid(o[:, i * self.T:(i + 1) * self.T], lay)
            place = p if place is None else place + p
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
