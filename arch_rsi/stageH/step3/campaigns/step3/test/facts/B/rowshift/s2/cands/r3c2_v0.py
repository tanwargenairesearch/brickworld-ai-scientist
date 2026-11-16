import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 48
PROJ = 16
HID = 16
LAYOUTS = [(False, False, False), (True, False, False), (False, True, False), (False, False, True)]


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.X = int(spec.X)
        self.Y = int(spec.Y)
        self.T = int(spec.n_types)
        self.P = int(spec.n_place)
        self.L = len(LAYOUTS)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = spec.in_channels + self.L * self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in range(self.L * self.T)])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in range(self.L * self.T)])
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)
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

    def pool_feats(self, p, wy, wx, fy, fx):
        if wy * wx == 1:
            return torch.cat([p, p, p], dim=1)
        pt = wy - 1 if fy else 0
        pb = 0 if fy else wy - 1
        pl = wx - 1 if fx else 0
        pr = 0 if fx else wx - 1
        q = F.pad(p, (pl, pr, pt, pb))
        mx = F.max_pool2d(q, (wy, wx), stride=1)
        mn = -F.max_pool2d(-q, (wy, wx), stride=1)
        av = F.avg_pool2d(q, (wy, wx), stride=1)
        return torch.cat([mx, mn, av], dim=1)

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
        p = F.relu(self.proj(f))
        place = None
        k = 0
        for lay in LAYOUTS:
            tr, fy, fx = lay[0], lay[1], lay[2]
            maps = []
            for t in range(self.T):
                dy, dx = self.fps[t]
                if tr:
                    wy, wx = dx, dy
                else:
                    wy, wx = dy, dx
                z = self.pool_feats(p, wy, wx, fy, fx)
                hh = F.relu(self.mix[k](z))
                maps.append(self.outs[k](hh))
                k += 1
            o = torch.cat(maps, dim=1)
            part = self.from_grid(o, tr, fy, fx)
            place = part if place is None else place + part
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
