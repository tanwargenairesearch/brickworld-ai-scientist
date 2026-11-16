import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 48
WIDTH = 40
GH = 64
AXES = ('t', 'y', 'x')
PERMS = [('t', 'y', 'x'), ('t', 'x', 'y'), ('y', 't', 'x'), ('x', 't', 'y'), ('y', 'x', 't'), ('x', 'y', 't')]


def make_layouts():
    res = []
    for p in PERMS:
        for fy in (False, True):
            for fx in (False, True):
                res.append((p, fy, fx))
    return res


def to_grid(v, sizes, lay):
    perm, fy, fx = lay
    m = v.reshape([v.shape[0]] + [sizes[a] for a in perm])
    order = [0] + [1 + perm.index(a) for a in AXES]
    m = m.permute(order)
    if fy:
        m = torch.flip(m, [2])
    if fx:
        m = torch.flip(m, [3])
    return m


def from_grid(m, lay):
    perm, fy, fx = lay
    if fy:
        m = torch.flip(m, [2])
    if fx:
        m = torch.flip(m, [3])
    order = [0] + [1 + AXES.index(a) for a in perm]
    return m.permute(order).reshape(m.shape[0], -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.t = int(spec.n_types)
        self.y = int(spec.Y)
        self.x = int(spec.X)
        self.n_place = int(spec.n_place)
        self.sizes = {'t': self.t, 'y': self.y, 'x': self.x}
        self.layouts = make_layouts()
        self.nl = len(self.layouts)
        cin = spec.in_channels + 2 * self.nl * self.t + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c3 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.out0 = nn.Conv2d(WIDTH, self.nl * self.t, 1)
        self.out1 = nn.Conv2d(WIDTH, self.nl * self.t, 1)
        for o in (self.out0, self.out1):
            nn.init.zeros_(o.weight)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done0 = nn.Linear(GH, 1)
        self.done1 = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def remap(self, o):
        place = None
        for i, lay in enumerate(self.layouts):
            p = from_grid(o[:, i * self.t:(i + 1) * self.t], lay)
            place = p if place is None else place + p
        return place

    def forward(self, obs, mask):
        b = obs.shape[0]
        mf = mask.to(obs.dtype)
        m0 = mf[:, :self.n_place]
        m1 = mf[:, 1:self.n_place + 1]
        ms = [to_grid(m0, self.sizes, lay) for lay in self.layouts] + [to_grid(m1, self.sizes, lay) for lay in self.layouts]
        yy = torch.linspace(-1.0, 1.0, self.y, device=obs.device, dtype=obs.dtype).view(1, 1, self.y, 1).expand(b, 1, self.y, self.x)
        xx = torch.linspace(-1.0, 1.0, self.x, device=obs.device, dtype=obs.dtype).view(1, 1, 1, self.x).expand(b, 1, self.y, self.x)
        h = torch.cat([obs] + ms + [yy, xx], dim=1)
        s = self.stem(h)
        z = F.relu(s + self.c1(s))
        z = F.relu(z + self.c2(z))
        f = F.relu(z + self.c3(z))
        p0 = self.remap(self.out0(f))
        p1 = self.remap(self.out1(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        zp = torch.zeros(b, self.n_place, device=obs.device, dtype=obs.dtype)
        logits = F.pad(p0, (0, 1)) + F.pad(p1, (1, 0))
        logits = logits + torch.cat([zp, self.done0(g)], dim=1) + torch.cat([self.done1(g), zp], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
