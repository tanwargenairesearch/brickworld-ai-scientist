import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
GH = 64
SPATIAL = ["id", "fy", "fx", "tr"]
OFFSETS = [0, 1]


def fwd_t(t, name):
    if name == "fy":
        return torch.flip(t, [2])
    if name == "fx":
        return torch.flip(t, [3])
    if name == "tr":
        return t.transpose(2, 3)
    return t


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        sq = self.Y == self.X
        self.spatial = [s for s in SPATIAL if sq or s != "tr"]
        self.combos = [(o, s) for o in OFFSETS for s in self.spatial]
        L = len(self.combos)
        cin = int(spec.in_channels) + L * self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.out = nn.Conv2d(WIDTH, L * self.T, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done_last = nn.Linear(GH, 1)
        self.done_first = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        P = self.n_place
        mf = mask.to(obs.dtype)
        ms = [mf[:, 0:P].reshape(B, self.T, self.Y, self.X),
              mf[:, 1:P + 1].reshape(B, self.T, self.Y, self.X)]
        ys = torch.linspace(-1.0, 1.0, self.Y, device=obs.device, dtype=obs.dtype)
        xs = torch.linspace(-1.0, 1.0, self.X, device=obs.device, dtype=obs.dtype)
        ys = ys.view(1, 1, self.Y, 1).expand(B, 1, self.Y, self.X)
        xs = xs.view(1, 1, 1, self.X).expand(B, 1, self.Y, self.X)
        feats = [obs, ys, xs]
        for off, s in self.combos:
            feats.append(fwd_t(ms[off], s))
        h = self.stem(torch.cat(feats, dim=1))
        x = F.relu(self.c1(h))
        f = F.relu(x + self.c2(x))
        o = self.out(f)
        places = [None, None]
        for k, (off, s) in enumerate(self.combos):
            v = fwd_t(o[:, k * self.T:(k + 1) * self.T], s).reshape(B, -1)
            places[off] = v if places[off] is None else places[off] + v
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        la = torch.cat([places[0], self.done_last(g)], dim=1)
        lb = torch.cat([self.done_first(g), places[1]], dim=1)
        logits = la + lb
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
