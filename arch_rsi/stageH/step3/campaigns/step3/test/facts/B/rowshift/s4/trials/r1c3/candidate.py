import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 40
WIDTH = 40
GH = 64
LAYOUTS = [(False, False, False), (True, False, False), (False, True, False), (False, False, True)]


def to_grid(v, t, y, x, lay):
    tr, fa, fb = lay
    if tr:
        m = v.reshape(v.shape[0], t, x, y)
    else:
        m = v.reshape(v.shape[0], t, y, x)
    if fa:
        m = torch.flip(m, [2])
    if fb:
        m = torch.flip(m, [3])
    if tr:
        m = m.permute(0, 1, 3, 2)
    return m


def from_grid(m, lay):
    tr, fa, fb = lay
    if tr:
        m = m.permute(0, 1, 3, 2)
    if fa:
        m = torch.flip(m, [2])
    if fb:
        m = torch.flip(m, [3])
    return m.reshape(m.shape[0], -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.t = int(spec.n_types)
        self.y = int(spec.Y)
        self.x = int(spec.X)
        self.n_place = int(spec.n_place)
        cin = spec.in_channels + len(LAYOUTS) * self.t + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, STEM, 1), nn.ReLU())
        self.c1 = nn.Conv2d(STEM, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.gctx = nn.Linear(2 * WIDTH, WIDTH)
        self.c3 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.outs = nn.ModuleList([nn.Conv2d(WIDTH, self.t, 1) for _ in LAYOUTS])
        for o in self.outs:
            nn.init.zeros_(o.weight)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        b = obs.shape[0]
        mp = mask[:, :self.n_place].to(obs.dtype)
        ms = [to_grid(mp, self.t, self.y, self.x, lay) for lay in LAYOUTS]
        yy = torch.linspace(-1.0, 1.0, self.y, device=obs.device, dtype=obs.dtype).view(1, 1, self.y, 1).expand(b, 1, self.y, self.x)
        xx = torch.linspace(-1.0, 1.0, self.x, device=obs.device, dtype=obs.dtype).view(1, 1, 1, self.x).expand(b, 1, self.y, self.x)
        h = torch.cat([obs] + ms + [yy, xx], dim=1)
        s = self.stem(h)
        z = F.relu(self.c1(s))
        z = F.relu(z + self.c2(z))
        gc = self.gctx(torch.cat([z.mean((2, 3)), z.amax((2, 3))], dim=1))
        z = z + F.relu(gc)[:, :, None, None]
        f = F.relu(z + self.c3(z))
        place = None
        for o, lay in zip(self.outs, LAYOUTS):
            p = from_grid(o(f), lay)
            place = p if place is None else place + p
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
