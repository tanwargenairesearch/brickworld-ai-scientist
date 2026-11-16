import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2}

STEM = 64
COL = 48
WIDTH = 48
NBLOCKS = 3
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
        cin = spec.in_channels + L * self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(NBLOCKS)])
        self.gproj = nn.Linear(2 * WIDTH, WIDTH)
        self.out = nn.Conv2d(WIDTH, L * self.T, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T = self.Y, self.X, self.T
        mp = mask[:, :self.n_place].to(obs.dtype)
        ms = [lay_in(mp, T, Y, X, o, k) for (o, k) in self.layouts]
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        h = torch.cat([obs] + ms + [yy, xx], dim=1)
        s = self.stem(h)
        x = F.relu(self.c1(s))
        for i, conv in enumerate(self.blocks):
            if i == 1:
                gv = F.relu(self.gproj(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)))
                x = x + gv[:, :, None, None]
            x = F.relu(x + conv(x))
        f = x
        o = self.out(f)
        place = None
        for i, (od, k) in enumerate(self.layouts):
            l = lay_out(o[:, i * T:(i + 1) * T], od, k)
            place = l if place is None else place + l
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
