import torch
import torch.nn as nn
import torch.nn.functional as F

TW = 32
LW = 32
P = 16
H = 16
GH = 64


def make_windows(dy, dx):
    if dy * dx == 1:
        return []
    wins = [(dy, dx)]
    if (dx, dy) != (dy, dx):
        wins.append((dx, dy))
    res = []
    for wy, wx in wins:
        res.append((wy, wx, False))
        res.append((wy, wx, True))
    return res


def pool_multi(q, wins):
    if len(wins) == 0:
        return q
    outs = []
    for wy, wx, br in wins:
        if br:
            r = F.pad(q, (wx - 1, 0, wy - 1, 0))
        else:
            r = F.pad(q, (0, wx - 1, 0, wy - 1))
        outs.append(F.max_pool2d(r, (wy, wx), stride=1))
        outs.append(-F.max_pool2d(-r, (wy, wx), stride=1))
    return torch.cat(outs, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.t = int(spec.n_types)
        self.y = int(spec.Y)
        self.x = int(spec.X)
        self.z = int(spec.Z)
        self.n_place = int(spec.n_place)
        self.wins = [make_windows(int(a), int(b)) for a, b in spec.footprints]
        cin = spec.in_channels + self.t + 2
        self.stem = nn.Conv2d(cin, 48, 1)
        self.c1 = nn.Conv2d(48, TW, 3, padding=1)
        self.c2 = nn.Conv2d(TW, TW, 3, padding=1)
        self.direct = nn.Conv2d(TW, self.t, 1)
        nn.init.zeros_(self.direct.weight)
        nn.init.zeros_(self.direct.bias)
        cl = 12 + 3 + 2 + self.t + self.z
        self.lin = nn.Conv2d(cl, LW, 1)
        self.tproj = nn.Conv2d(TW, LW, 1)
        self.l2 = nn.Conv2d(LW, LW, 1)
        self.proj = nn.Conv2d(LW, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(max(1, 2 * len(w)) * P, H, 1) for w in self.wins])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.wins])
        self.glob = nn.Sequential(nn.Linear(2 * TW, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        b = obs.shape[0]
        Z, Y, X = self.z, self.y, self.x
        mp = mask[:, :self.n_place].to(obs.dtype).reshape(b, self.t, Y, X)
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(b, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(b, 1, Y, X)
        s = F.relu(self.stem(torch.cat([obs, mp, yy, xx], dim=1)))
        h = F.relu(self.c1(s))
        f = F.relu(h + self.c2(h))
        build = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        grouped = torch.stack([build, tgt, miss, extra], dim=2)
        inter = obs[:, :4 * Z].reshape(b, Z, 4, Y, X)
        bp = F.pad(build, (0, 0, 0, 0, 1, 1))
        mpd = F.pad(miss, (0, 0, 0, 0, 1, 1))
        neigh = torch.stack([bp[:, :Z], bp[:, 2:], mpd[:, :Z], mpd[:, 2:]], dim=2)
        shared = torch.cat([obs[:, 4 * Z:4 * Z + 3], yy, xx, mp], dim=1)
        shared = shared.unsqueeze(1).expand(b, Z, shared.shape[1], Y, X)
        zid = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(b, Z, Z, Y, X)
        L = torch.cat([grouped, inter, neigh, shared, zid], dim=2).reshape(b * Z, -1, Y, X)
        tf = self.tproj(f).unsqueeze(1).expand(b, Z, LW, Y, X).reshape(b * Z, LW, Y, X)
        u = F.relu(self.lin(L) + tf)
        u = F.relu(self.l2(u))
        q = self.proj(u)
        sc = []
        for w, mix, out in zip(self.wins, self.mix, self.outs):
            sc.append(out(F.relu(mix(pool_multi(q, w)))))
        sc = torch.cat(sc, dim=1).reshape(b, Z, self.t, Y, X)
        place = torch.logsumexp(sc, dim=1) + self.direct(f)
        place = place.reshape(b, -1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
