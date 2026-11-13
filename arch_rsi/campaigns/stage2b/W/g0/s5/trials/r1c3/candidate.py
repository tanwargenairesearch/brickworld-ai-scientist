"""Layer-factorised encoder: a shared 2D conv processes every layer z with its local context (build at z,
below and above, target, missing, extra, heights, time, z position); per-layer features are concatenated,
mixed by a 1x1 conv, and scored by a footprint-pooled placement head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

LW = 24
WIDTH = 64
PW = 32
K_IN = 10


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
        self.layer_enc = nn.Sequential(
            nn.Conv2d(K_IN, LW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(LW, LW, 3, padding=1), nn.ReLU(),
        )
        self.mix = nn.Sequential(nn.Conv2d(self.Z * LW, WIDTH, 1), nn.ReLU())
        self.proj = nn.Conv2d(WIDTH, PW, 1)
        self.type_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * PW, PW, 1), nn.ReLU(), nn.Conv2d(PW, 1, 1))
            for _ in self.fps
        ])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        build = obs[:, 0:Z]
        target = obs[:, Z:2 * Z]
        missing = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        below = F.pad(build[:, :Z - 1], (0, 0, 0, 0, 1, 0), value=1.0)
        above = F.pad(build[:, 1:], (0, 0, 0, 0, 0, 1), value=0.0)
        per_layer = torch.stack([build, below, above, target, missing, extra], dim=2)
        glob_ch = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zpos = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1)).view(1, Z, 1, 1, 1)
        zpos = zpos.expand(B, Z, 1, Y, X)
        x = torch.cat([per_layer, glob_ch, zpos], dim=2).reshape(B * Z, K_IN, Y, X)
        h = self.layer_enc(x).reshape(B, Z * LW, Y, X)
        f = self.mix(h)
        p = self.proj(f)
        outs = []
        for (dy, dx), head in zip(self.fps, self.type_heads):
            outs.append(head(fp_pool(p, dy, dx)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
