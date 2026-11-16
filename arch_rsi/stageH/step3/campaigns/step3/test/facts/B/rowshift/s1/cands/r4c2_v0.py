import math
import torch
import torch.nn as nn
import torch.nn.functional as F

W = 24
SH = 32
LAYOUTS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.Z = int(spec.Z)
        self.n_place = int(spec.n_place)
        self.layouts = list(LAYOUTS)
        L = len(self.layouts)
        cin = 9 + spec.in_channels + self.Z + L * self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, SH, 1), nn.ReLU(),
            nn.Conv2d(SH, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.out = nn.Conv2d(W, L * self.T, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        self.logz = math.log(max(self.Z, 1))

    def forward(self, obs, mask):
        B = obs.shape[0]
        T, Y, X, Z = self.T, self.Y, self.X, self.Z
        bld = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        mis = obs[:, 2 * Z:3 * Z]
        ext = obs[:, 3 * Z:4 * Z]
        bld_lo = F.pad(bld, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        tgt_lo = F.pad(tgt, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        mis_lo = F.pad(mis, (0, 0, 0, 0, 1, 0), value=0.0)[:, :Z]
        bld_hi = F.pad(bld, (0, 0, 0, 0, 0, 1), value=0.0)[:, 1:]
        tgt_hi = F.pad(tgt, (0, 0, 0, 0, 0, 1), value=0.0)[:, 1:]
        per = torch.stack([bld, tgt, mis, ext, bld_lo, tgt_lo, mis_lo, bld_hi, tgt_hi], dim=2)
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
        shared = torch.cat(ins, dim=1)
        Cs = shared.shape[1]
        shared = shared.unsqueeze(1).expand(B, Z, Cs, Y, X)
        zoh = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        inp = torch.cat([per, shared, zoh], dim=2)
        inp = inp.reshape(B * Z, inp.shape[2], Y, X)
        s = self.stem(inp)
        h = F.relu(self.c1(s))
        h = F.relu(h + self.c2(h))
        o = self.out(h)
        o = o.view(B, Z, o.shape[1], Y, X)
        o = torch.logsumexp(o, dim=1) - self.logz
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
        hz = h.view(B, Z, h.shape[1], Y, X)
        g = self.glob(torch.cat([hz.mean((1, 3, 4)), hz.amax(dim=4).amax(dim=3).amax(dim=1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
