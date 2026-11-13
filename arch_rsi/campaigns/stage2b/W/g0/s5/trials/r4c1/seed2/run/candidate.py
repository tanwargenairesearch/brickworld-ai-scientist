"""r2c3 (per-layer footprint scoring) with a lower entropy bonus (ent_coef 0.003)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

LW = 20
PW = 16
HH = 16
GW = 64

RECIPE = {"ent_coef": 0.003}


def fp_pool(p, dy, dx):
    if dy == 1 and dx == 1:
        return torch.cat([p, p, p], dim=1)
    pad = (0, dx - 1, 0, dy - 1)
    mx = F.max_pool2d(F.pad(p, pad, value=-1e4), (dy, dx), stride=1)
    mn = -F.max_pool2d(F.pad(-p, pad, value=-1e4), (dy, dx), stride=1)
    av = F.avg_pool2d(F.pad(p, pad), (dy, dx), stride=1)
    return torch.cat([mx, mn, av], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.k_in = 2 * self.Z + 7
        self.enc = nn.Sequential(
            nn.Conv2d(self.k_in, LW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(LW, LW, 3, padding=1), nn.ReLU(),
        )
        self.proj = nn.Conv2d(LW, PW, 1)
        self.type_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * PW, HH, 1), nn.ReLU(), nn.Conv2d(HH, 1, 1))
            for _ in self.fps
        ])
        self.glob = nn.Sequential(nn.Linear(2 * LW, GW), nn.ReLU())
        self.done_head = nn.Linear(GW, 1)
        self.value_head = nn.Linear(GW, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        build = obs[:, 0:Z]
        target = obs[:, Z:2 * Z]
        missing = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        g3 = obs[:, 4 * Z:4 * Z + 3]
        bp = F.pad(build, (0, 0, 0, 0, 1, 0), value=1.0)
        bp = F.pad(bp, (0, 0, 0, 0, 0, Z), value=0.0)
        mp = F.pad(missing, (0, 0, 0, 0, Z - 1, 0), value=0.0)
        rows = []
        for z in range(Z):
            zpos = obs.new_full((B, 1, Y, X), float(z) / max(Z - 1, 1))
            rows.append(torch.cat([
                bp[:, z:z + Z + 1],
                mp[:, z:z + Z - 1],
                target[:, z:z + 1],
                missing[:, z:z + 1],
                extra[:, z:z + 1],
                g3,
                zpos,
            ], dim=1))
        x = torch.stack(rows, dim=1).reshape(B * Z, self.k_in, Y, X)
        e = self.enc(x)
        p = self.proj(e)
        outs = []
        for (dy, dx), head in zip(self.fps, self.type_heads):
            outs.append(head(fp_pool(p, dy, dx)))
        s = torch.cat(outs, dim=1).reshape(B, Z, len(self.fps), Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        eg = e.reshape(B, Z, LW, Y, X)
        g = self.glob(torch.cat([eg.mean((1, 3, 4)), eg.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
