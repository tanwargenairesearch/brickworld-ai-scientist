import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 48
WIDTH = 40
GH = 64
AXES = ('t', 'y', 'x')
PERMS = [('t', 'y', 'x'), ('t', 'x', 'y'), ('y', 't', 'x'), ('x', 't', 'y'), ('y', 'x', 't'), ('x', 'y', 't')]
LAYOUTS = [(p, fy, fx) for p in PERMS for fy in (False, True) for fx in (False, True)]


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
        self.nl = len(LAYOUTS)
        cin = spec.in_channels + self.nl * self.t + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c3 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.out = nn.Conv2d(WIDTH, self.nl * self.t, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        b = obs.shape[0]
        mp = mask[:, :self.n_place].to(obs.dtype)
        ms = [to_grid(mp, self.sizes, lay) for lay in LAYOUTS]
        yy = torch.linspace(-1.0, 1.0, self.y, device=obs.device, dtype=obs.dtype).view(1, 1, self.y, 1).expand(b, 1, self.y, self.x)
        xx = torch.linspace(-1.0, 1.0, self.x, device=obs.device, dtype=obs.dtype).view(1, 1, 1, self.x).expand(b, 1, self.y, self.x)
        h = torch.cat([obs] + ms + [yy, xx], dim=1)
        s = self.stem(h)
        z = F.relu(s + self.c1(s))
        z = F.relu(z + self.c2(z))
        f = F.relu(z + self.c3(z))
        o = self.out(f)
        place = None
        for i, lay in enumerate(LAYOUTS):
            p = from_grid(o[:, i * self.t:(i + 1) * self.t], lay)
            place = p if place is None else place + p
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
