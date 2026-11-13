"""Per-layer token features with footprint max/min pooling at every layer, combined by log-sum-exp over layers."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
LW = 32
PW = 24
HID = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        W = WIDTH
        self.Z = spec.Z
        self.Y = spec.Y
        self.X = spec.X
        self.n_types = spec.n_types
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Sequential(nn.Conv2d(spec.in_channels, W, 1), nn.ReLU())
        self.ctx = nn.Conv2d(W, W, 3, padding=1)
        self.bcast = nn.Linear(2 * W, W)
        self.lay_in = nn.Conv2d(8 + spec.Z, LW, 1)
        self.lay_f = nn.Conv2d(W, LW, 1)
        self.lay2 = nn.Conv2d(LW, PW, 1)
        self.mix = nn.Conv2d(2 * PW, HID, 1)
        self.type_bias = nn.Parameter(torch.zeros(spec.n_types, HID, 1, 1))
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in range(spec.n_types)])
        self.head2d = nn.Conv2d(W, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, W), nn.ReLU())
        self.done_head = nn.Linear(W, 1)
        self.value_head = nn.Linear(W, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Z, Y, X = self.Z, self.Y, self.X
        h = self.stem(obs)
        f = h + F.relu(self.ctx(h))
        g0 = F.relu(self.bcast(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)))
        f = f + g0[:, :, None, None]

        build = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        below = F.pad(build, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        per = torch.stack([build, tgt, miss, extra, below], dim=2)
        g3 = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zid = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        tok = torch.cat([per, g3, zid], dim=2).reshape(B * Z, 8 + Z, Y, X)
        fz = self.lay_f(f).unsqueeze(1).expand(B, Z, LW, Y, X).reshape(B * Z, LW, Y, X)
        u = F.relu(self.lay_in(tok) + fz)
        u = F.relu(self.lay2(u))

        scores = []
        for t, (dy, dx) in enumerate(self.footprints):
            pp = F.pad(u, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(pp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-pp, (dy, dx), stride=1)
            zz = F.relu(self.mix(torch.cat([mx, mn], dim=1)) + self.type_bias[t])
            s = self.outs[t](zz).reshape(B, Z, Y * X)
            scores.append(torch.logsumexp(s, dim=1))
        place = torch.stack(scores, dim=1) + self.head2d(f).flatten(2)
        place = place.flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
