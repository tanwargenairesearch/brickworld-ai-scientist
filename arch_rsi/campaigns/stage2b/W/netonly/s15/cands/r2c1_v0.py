"""Layer-factorised policy (as r1c3) with a nonlinear per-type footprint head producing score and layer gate."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 18
HEAD = 12
GLOB = 64


def footprint_pool(f, dy, dx):
    if dy == 1 and dx == 1:
        return torch.cat([f, f, f], dim=1)
    p = F.pad(f, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(p, (dy, dx), stride=1)
    mn = -F.max_pool2d(-p, (dy, dx), stride=1)
    av = F.avg_pool2d(p, (dy, dx), stride=1)
    return torch.cat([mx, mn, av], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        c_in = 15 + self.Z
        self.body = nn.Sequential(
            nn.Conv2d(c_in, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU(),
        )
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.place_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * WIDTH, HEAD, 1), nn.ReLU(), nn.Conv2d(HEAD, 2, 1))
            for _ in self.fps
        ])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        per = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], dim=2)
        zeros = torch.zeros(B, 1, 4, Y, X, device=obs.device, dtype=obs.dtype)
        below = torch.cat([zeros, per[:, :Z - 1]], dim=1)
        above = torch.cat([per[:, 1:], zeros], dim=1)
        g = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zemb = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        x = torch.cat([below, per, above, g, zemb], dim=2).reshape(B * Z, 15 + Z, Y, X)
        h = self.body(x)
        outs = []
        for head, fp in zip(self.place_heads, self.fps):
            o = head(footprint_pool(h, fp[0], fp[1])).view(B, Z, 2, Y, X)
            w = torch.softmax(o[:, :, 1], dim=1)
            outs.append((w * o[:, :, 0]).sum(1))
        place = torch.stack(outs, dim=1).flatten(1)
        hv = h.view(B, Z, WIDTH, Y, X)
        pooled = torch.cat([hv.mean((1, 3, 4)), hv.amax((1, 3, 4))], dim=1)
        gg = self.glob(pooled)
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
