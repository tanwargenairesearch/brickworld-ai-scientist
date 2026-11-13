"""Layer-shared 2D encoder (per-layer planes + neighbours), stacked and mixed, then context + footprint heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F

LW = 16
WIDTH = 40
HID = 24


def fp_pool(f, dy, dx):
    p = F.pad(f, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(p, (dy, dx), stride=1)
    mn = -F.max_pool2d(-p, (dy, dx), stride=1)
    return mx, mn


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        n_in = 4 * 3 + 3 + 1
        self.layer_body = nn.Sequential(
            nn.Conv2d(n_in, LW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(LW, LW, 3, padding=1), nn.ReLU())
        self.mix = nn.Sequential(
            nn.Conv2d(self.Z * LW, WIDTH, 1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.ctx = nn.Linear(WIDTH, WIDTH)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * WIDTH, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in self.fps])
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        per = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        below = F.pad(per, (0, 0, 0, 0, 1, 0))[:, :, :Z]
        above = F.pad(per, (0, 0, 0, 0, 0, 1))[:, :, 1:]
        g3 = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zpos = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1))
        zpos = zpos.view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        x = torch.cat([per, below, above, g3, zpos], 1)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * Z, -1, Y, X)
        h = self.layer_body(x).reshape(B, Z * LW, Y, X)
        f = self.mix(h)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        hc = F.relu(f + self.ctx(g)[:, :, None, None])
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            mx, mn = fp_pool(hc, dy, dx)
            outs.append(head(torch.cat([hc, mx, mn], 1)).flatten(1))
        logits = torch.cat(outs + [self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
