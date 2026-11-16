import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 48
HID = 32
LAYOUTS = [(False, False, False), (True, False, False), (False, True, False), (False, False, True)]


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.X = int(spec.X)
        self.Y = int(spec.Y)
        self.T = int(spec.n_types)
        self.P = int(spec.n_place)
        self.L = len(LAYOUTS)
        cin = spec.in_channels + self.L * self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hmix = nn.Conv2d(WIDTH, HID, 1)
        self.hout = nn.Conv2d(HID, self.L * self.T, 1)
        nn.init.normal_(self.hout.weight, std=0.01)
        nn.init.zeros_(self.hout.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def to_grid(self, m, tr, fy, fx):
        b = m.shape[0]
        if tr:
            g = m.reshape(b, self.T, self.X, self.Y).transpose(2, 3)
        else:
            g = m.reshape(b, self.T, self.Y, self.X)
        dims = []
        if fy:
            dims.append(2)
        if fx:
            dims.append(3)
        if len(dims) > 0:
            g = g.flip(dims)
        return g

    def from_grid(self, g, tr, fy, fx):
        b = g.shape[0]
        dims = []
        if fy:
            dims.append(2)
        if fx:
            dims.append(3)
        if len(dims) > 0:
            g = g.flip(dims)
        if tr:
            g = g.transpose(2, 3)
        return g.reshape(b, -1)

    def forward(self, obs, mask):
        b = obs.shape[0]
        m = mask[:, :self.P].to(obs.dtype)
        grids = [self.to_grid(m, lay[0], lay[1], lay[2]) for lay in LAYOUTS]
        ys = torch.linspace(-1.0, 1.0, self.Y, device=obs.device, dtype=obs.dtype).view(1, 1, self.Y, 1).expand(b, 1, self.Y, self.X)
        xs = torch.linspace(-1.0, 1.0, self.X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, self.X).expand(b, 1, self.Y, self.X)
        x = torch.cat([obs] + grids + [ys, xs], dim=1)
        s = self.stem(x)
        h = F.relu(self.c1(s))
        f = F.relu(h + self.c2(h))
        o = self.hout(F.relu(self.hmix(f)))
        place = None
        for i, lay in enumerate(LAYOUTS):
            part = self.from_grid(o[:, i * self.T:(i + 1) * self.T], lay[0], lay[1], lay[2])
            place = part if place is None else place + part
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
