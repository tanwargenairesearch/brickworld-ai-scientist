import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
HID = 32
LAYOUTS = [(0, ()), (0, (2,)), (0, (3,)), (0, (2, 3)),
           (1, ()), (1, (2,)), (1, (3,)), (1, (2, 3))]
SHIFTS = [0, 1]


def to_frame(m, k, T, Y, X):
    B = m.shape[0]
    if k == 0:
        return m.reshape(B, T, Y, X)
    if k == 1:
        return m.reshape(B, T, X, Y).permute(0, 1, 3, 2)
    if k == 2:
        return m.reshape(B, Y, X, T).permute(0, 3, 1, 2)
    return m.reshape(B, X, Y, T).permute(0, 3, 2, 1)


def from_frame(g, k):
    B = g.shape[0]
    if k == 0:
        return g.reshape(B, -1)
    if k == 1:
        return g.permute(0, 1, 3, 2).reshape(B, -1)
    if k == 2:
        return g.permute(0, 2, 3, 1).reshape(B, -1)
    return g.permute(0, 3, 2, 1).reshape(B, -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T, self.Y, self.X = spec.n_types, spec.Y, spec.X
        self.n_place = spec.n_place
        self.n_actions = spec.n_actions
        L = len(LAYOUTS)
        S = len(SHIFTS)
        cin = spec.in_channels + S * L * self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hid = nn.Conv2d(WIDTH, HID, 1)
        self.out = nn.Conv2d(HID, S * L * self.T, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, S)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        T, Y, X = self.T, self.Y, self.X
        NP, NA = self.n_place, self.n_actions
        mf = mask.float()
        chans = []
        for s in SHIFTS:
            ms = mf[:, s:s + NP]
            for k, fl in LAYOUTS:
                g = to_frame(ms, k, T, Y, X)
                if len(fl) > 0:
                    g = g.flip(fl)
                chans.append(g)
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device).view(1, 1, 1, X).expand(B, 1, Y, X)
        s0 = self.stem(torch.cat([obs] + chans + [yy, xx], dim=1))
        x = F.relu(self.c1(s0))
        f = F.relu(x + self.c2(x))
        o = self.out(F.relu(self.hid(f)))
        o = o.reshape(B, len(SHIFTS), len(LAYOUTS), T, Y, X)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        d = self.done_head(g)
        logits = 0
        for si, s in enumerate(SHIFTS):
            total = 0
            for li, (k, fl) in enumerate(LAYOUTS):
                h = o[:, si, li]
                if len(fl) > 0:
                    h = h.flip(fl)
                total = total + from_frame(h, k)
            logits = logits + F.pad(total, (s, NA - NP - s))
            pos = NP if s == 0 else 0
            logits = logits + F.pad(d[:, si:si + 1], (pos, NA - 1 - pos))
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
