"""Layer-factorised policy with learned cross-layer cumulative-max features (below and above each layer)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 18
K = 6
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
        self.conv1 = nn.Conv2d(c_in, WIDTH, 3, padding=1)
        self.proj = nn.Conv2d(WIDTH, K, 1)
        self.conv2 = nn.Conv2d(WIDTH + 2 * K, WIDTH, 3, padding=1)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.place_heads = nn.ModuleList([nn.Conv2d(3 * WIDTH, 2, 1) for _ in self.fps])
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
        h1 = F.relu(self.conv1(x))
        p = F.relu(self.proj(h1)).view(B, Z, K, Y, X)
        pz = torch.zeros_like(p[:, :1])
        cb = torch.cummax(p, dim=1).values
        agg_below = torch.cat([pz, cb[:, :Z - 1]], dim=1)
        ca = torch.flip(torch.cummax(torch.flip(p, [1]), dim=1).values, [1])
        agg_above = torch.cat([ca[:, 1:], pz], dim=1)
        h2_in = torch.cat([h1, agg_below.reshape(B * Z, K, Y, X), agg_above.reshape(B * Z, K, Y, X)], dim=1)
        h = F.relu(self.conv2(h2_in))
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
