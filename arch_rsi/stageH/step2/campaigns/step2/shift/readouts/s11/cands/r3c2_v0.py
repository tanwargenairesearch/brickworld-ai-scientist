import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 32
LP = 16
LH = 16


def xroll(o, s):
    return torch.cat([o[..., s:], o[..., :s]], dim=3)


class LayerHead(nn.Module):
    def __init__(self, c, Z, n_sets, footprints):
        super().__init__()
        self.Z = int(Z)
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.inp = nn.Conv2d(4 * n_sets + self.Z, LP, 1)
        self.tp = nn.Conv2d(c, LP, 1)
        self.c2 = nn.Conv2d(LP, LP, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * LP, LH, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(LH, 1, 1) for _ in self.fps])

    def forward(self, parts, f):
        B, _, Y, X = f.shape
        Z = self.Z
        ls = torch.cat([p[:, :4 * Z].reshape(B, 4, Z, Y, X) for p in parts], dim=1)
        ls = ls.permute(0, 2, 1, 3, 4)
        eye = torch.eye(Z, device=f.device, dtype=f.dtype).reshape(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        ls = torch.cat([ls, eye], dim=2).reshape(B * Z, -1, Y, X)
        t = self.tp(f).unsqueeze(1).expand(B, Z, LP, Y, X).reshape(B * Z, LP, Y, X)
        h = F.relu(self.inp(ls) + t)
        p = F.relu(h + self.c2(h))
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            hh = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(hh))
        o = torch.cat(res, dim=1).reshape(B, Z, len(self.fps), Y, X)
        return torch.logsumexp(o, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        X = int(spec.X)
        sh = set([X // 2, X - X // 2])
        self.shifts = sorted([s for s in sh if 0 < s < X])
        n_sets = 1 + len(self.shifts)
        cin = spec.in_channels * n_sets
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = LayerHead(WIDTH, spec.Z, n_sets, spec.footprints)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        parts = [obs] + [xroll(obs, s) for s in self.shifts]
        s = self.stem(torch.cat(parts, dim=1))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(parts, f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
