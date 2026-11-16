import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
ORDERS = ["type", "cell"]
OFFSETS = [0, 1]
DIHEDRAL = [(False, False, False), (False, False, True), (False, True, False), (False, True, True),
            (True, False, False), (True, False, True), (True, True, False), (True, True, True)]


def make_perm(T, Y, X, order, tr, fy, fx):
    n = T * Y * X
    base = torch.arange(n)
    if order == "type":
        if tr:
            g = base.view(T, X, Y).permute(0, 2, 1)
        else:
            g = base.view(T, Y, X)
    else:
        if tr:
            g = base.view(X, Y, T).permute(2, 1, 0)
        else:
            g = base.view(Y, X, T).permute(2, 0, 1)
    if fy:
        g = g.flip(1)
    if fx:
        g = g.flip(2)
    perm = g.reshape(-1).clone()
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n)
    return perm, inv


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.X = int(spec.X)
        self.Y = int(spec.Y)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        perms = []
        invs = []
        for o in ORDERS:
            for (tr, fy, fx) in DIHEDRAL:
                p, q = make_perm(self.T, self.Y, self.X, o, tr, fy, fx)
                perms.append(p)
                invs.append(q)
        self.P = len(perms)
        self.register_buffer("perms", torch.stack(perms, 0))
        self.register_buffer("invs", torch.stack(invs, 0))
        self.offsets = list(OFFSETS)
        L = self.P * len(self.offsets)
        cin = int(spec.in_channels) + L * self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.out = nn.Conv2d(WIDTH, L * self.T, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_heads = nn.ModuleList([nn.Linear(64, 1) for _ in self.offsets])
        self.value_head = nn.Linear(64, 1)
        ys = torch.linspace(-1.0, 1.0, self.Y).view(1, 1, self.Y, 1).expand(1, 1, self.Y, self.X)
        xs = torch.linspace(-1.0, 1.0, self.X).view(1, 1, 1, self.X).expand(1, 1, self.Y, self.X)
        self.register_buffer("coords", torch.cat([ys, xs], dim=1).contiguous())

    def forward(self, obs, mask):
        B = obs.shape[0]
        n = self.n_place
        mf = mask.to(obs.dtype)
        chans = [obs]
        flat_perm = self.perms.reshape(-1)
        for k in self.offsets:
            mk = mf[:, k:k + n]
            chans.append(mk[:, flat_perm].reshape(B, self.P * self.T, self.Y, self.X))
        chans.append(self.coords.expand(B, -1, -1, -1))
        x = torch.cat(chans, dim=1)
        s = self.stem(x)
        h = F.relu(self.c1(s))
        f = F.relu(h + self.c2(h))
        o = self.out(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        idx = self.invs.unsqueeze(0).expand(B, -1, -1)
        logits = None
        for j, k in enumerate(self.offsets):
            oj = o[:, j * self.P * self.T:(j + 1) * self.P * self.T].reshape(B, self.P, n)
            place = torch.gather(oj, 2, idx).sum(1)
            d = self.done_heads[j](g)
            if k == 0:
                lj = torch.cat([place, d], dim=1)
            else:
                lj = torch.cat([d, place], dim=1)
            logits = lj if logits is None else logits + lj
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
