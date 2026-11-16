import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
ORDERS = [0, 1, 2]
DIHEDRAL = [0, 1, 2, 3, 4, 5, 6, 7]


def lay_in(mp, T, Y, X, order, k):
    B = mp.shape[0]
    if k >= 4:
        A, C = X, Y
    else:
        A, C = Y, X
    if order == 0:
        m = mp.reshape(B, T, A, C)
    elif order == 1:
        m = mp.reshape(B, A, C, T).permute(0, 3, 1, 2)
    else:
        m = mp.reshape(B, A, T, C).permute(0, 2, 1, 3)
    if k >= 4:
        m = m.transpose(2, 3)
    if k & 1:
        m = m.flip(2)
    if k & 2:
        m = m.flip(3)
    return m


def lay_out(o, order, k):
    B = o.shape[0]
    if k & 1:
        o = o.flip(2)
    if k & 2:
        o = o.flip(3)
    if k >= 4:
        o = o.transpose(2, 3)
    if order == 1:
        o = o.permute(0, 2, 3, 1)
    elif order == 2:
        o = o.permute(0, 2, 1, 3)
    return o.reshape(B, -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.layouts = [(o, k) for o in ORDERS for k in DIHEDRAL]
        L = len(self.layouts)
        cin = spec.in_channels + 2 * L * self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.out = nn.Conv2d(WIDTH, 2 * L * self.T, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 2)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T = self.Y, self.X, self.T
        P = self.n_place
        L = len(self.layouts)
        mf = mask.to(obs.dtype)
        m0 = mf[:, :P]
        m1 = mf[:, 1:P + 1]
        ms = [lay_in(m0, T, Y, X, o, k) for (o, k) in self.layouts]
        ms = ms + [lay_in(m1, T, Y, X, o, k) for (o, k) in self.layouts]
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        h = torch.cat([obs] + ms + [yy, xx], dim=1)
        s = self.stem(h)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        o = self.out(f)
        p0 = None
        p1 = None
        for i, (od, k) in enumerate(self.layouts):
            l0 = lay_out(o[:, i * T:(i + 1) * T], od, k)
            j = L + i
            l1 = lay_out(o[:, j * T:(j + 1) * T], od, k)
            p0 = l0 if p0 is None else p0 + l0
            p1 = l1 if p1 is None else p1 + l1
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        d = self.done_head(g)
        logits = torch.cat([p0, d[:, 0:1]], dim=1) + torch.cat([d[:, 1:2], p1], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
