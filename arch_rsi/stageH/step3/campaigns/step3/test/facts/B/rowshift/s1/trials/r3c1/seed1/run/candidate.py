import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
PROJ = 12
HID = 16
LAYOUTS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]


def to_action(a, tr, fy, fx):
    if fx:
        a = a.flip(3)
    if fy:
        a = a.flip(2)
    if tr:
        a = a.transpose(2, 3)
    return a


def pool_feats(p, dy, dx):
    if dy * dx == 1:
        return [p, p, p, p, p]
    out = [p]
    for a, b in ((dy, dx), (dx, dy)):
        q = F.pad(p, (0, b - 1, 0, a - 1))
        out.append(F.max_pool2d(q, (a, b), stride=1))
        out.append(-F.max_pool2d(-q, (a, b), stride=1))
    return out


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.layouts = list(LAYOUTS)
        L = len(self.layouts)
        T = self.T
        cin = spec.in_channels + L * T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c3 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.out = nn.Conv2d(WIDTH, L * T, 1)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.mixes = nn.ModuleList([nn.Conv2d(T * 5 * PROJ, T * HID, 1, groups=T) for _ in range(L)])
        self.outs = nn.ModuleList([nn.Conv2d(T * HID, T, 1, groups=T) for _ in range(L)])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        T, Y, X = self.T, self.Y, self.X
        m = mask[:, :self.n_place].float()
        ins = [obs]
        for tr, fy, fx in self.layouts:
            if tr:
                a = m.reshape(B, T, X, Y).transpose(2, 3)
            else:
                a = m.reshape(B, T, Y, X)
            if fy:
                a = a.flip(2)
            if fx:
                a = a.flip(3)
            ins.append(a)
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device).view(1, 1, 1, X).expand(B, 1, Y, X)
        ins.append(yy)
        ins.append(xx)
        s = self.stem(torch.cat(ins, dim=1))
        h = F.relu(self.c1(s))
        h = F.relu(h + self.c2(h))
        h = F.relu(h + self.c3(h))
        q = self.out(h)
        p = self.proj(h)
        place = None
        for k, (tr, fy, fx) in enumerate(self.layouts):
            pk = to_action(p, tr, fy, fx)
            feats = []
            for dy, dx in self.fps:
                feats.extend(pool_feats(pk, dy, dx))
            f = torch.cat(feats, dim=1)
            z = self.outs[k](F.relu(self.mixes[k](f)))
            z = z + to_action(q[:, k * T:(k + 1) * T], tr, fy, fx)
            z = z.reshape(B, -1)
            place = z if place is None else place + z
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
