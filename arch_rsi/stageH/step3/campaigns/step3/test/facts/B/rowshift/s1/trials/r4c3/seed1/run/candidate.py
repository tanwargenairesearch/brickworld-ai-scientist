import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
DW = 32
LAYOUTS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        self.layouts = list(LAYOUTS)
        L = len(self.layouts)
        cin = spec.in_channels + L * self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c3 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.out = nn.Conv2d(WIDTH, L * self.T, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.dpath = nn.Sequential(
            nn.Conv2d(spec.in_channels, DW, 1), nn.ReLU(),
            nn.Conv2d(DW, DW, 1), nn.ReLU())
        self.dmix = nn.Sequential(nn.Linear(64 + 2 * DW, 64), nn.ReLU())
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
        o = self.out(h)
        place = None
        for k, (tr, fy, fx) in enumerate(self.layouts):
            a = o[:, k * T:(k + 1) * T]
            if fx:
                a = a.flip(3)
            if fy:
                a = a.flip(2)
            if tr:
                a = a.transpose(2, 3)
            a = a.reshape(B, -1)
            place = a if place is None else place + a
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        d = self.dpath(obs)
        dg = self.dmix(torch.cat([g, d.amax((2, 3)), d.mean((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(dg)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
