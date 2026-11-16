import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 48
COL = 40
WIDTH = 40
HID = 32
LAYOUTS = [(0, ()), (0, (2,)), (0, (3,)), (0, (2, 3)),
           (1, ()), (1, (2,)), (1, (3,)), (1, (2, 3))]


def to_layouts(m, T, Y, X):
    B = m.shape[0]
    bases = [m.reshape(B, T, Y, X), m.reshape(B, T, X, Y).permute(0, 1, 3, 2)]
    out = []
    for k, f in LAYOUTS:
        g = bases[k]
        if len(f) > 0:
            g = g.flip(f)
        out.append(g)
    return torch.cat(out, dim=1)


def from_layouts(o, T, Y, X):
    B = o.shape[0]
    o = o.reshape(B, len(LAYOUTS), T, Y, X)
    total = 0
    for i, (k, f) in enumerate(LAYOUTS):
        g = o[:, i]
        if len(f) > 0:
            g = g.flip(f)
        if k == 1:
            g = g.permute(0, 1, 3, 2)
        total = total + g.reshape(B, -1)
    return total


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T, self.Y, self.X = spec.n_types, spec.Y, spec.X
        self.n_place = spec.n_place
        L = len(LAYOUTS)
        cin = spec.in_channels + L * self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c3 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hid = nn.Conv2d(WIDTH, HID, 1)
        self.out = nn.Conv2d(HID, L * self.T, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        T, Y, X = self.T, self.Y, self.X
        m = mask[:, :self.n_place].float()
        mc = to_layouts(m, T, Y, X)
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device).view(1, 1, 1, X).expand(B, 1, Y, X)
        s = self.stem(torch.cat([obs, mc, yy, xx], dim=1))
        x = F.relu(self.c1(s))
        x = F.relu(x + self.c2(x))
        f = F.relu(x + self.c3(x))
        o = self.out(F.relu(self.hid(f)))
        place = from_layouts(o, T, Y, X)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
