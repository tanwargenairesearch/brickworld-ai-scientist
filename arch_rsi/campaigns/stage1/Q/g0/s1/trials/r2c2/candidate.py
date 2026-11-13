import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 16
GW = 64
EMB = 8
HID = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.n_types = spec.n_types
        self.fps = [(int(a), int(b)) for (a, b) in spec.footprints]
        T = spec.n_types
        self.in3 = nn.Conv3d(5, WIDTH, 3, padding=1)
        self.in2 = nn.Conv2d(3, WIDTH, 1)
        self.c2 = nn.Conv3d(WIDTH, WIDTH, 3, padding=1)
        self.cell = nn.Conv3d(WIDTH, T * EMB, 1)
        self.mix = nn.Conv3d(T * 3 * EMB, T * HID, 1, groups=T)
        self.out = nn.Conv3d(T * HID, T, 1, groups=T)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GW), nn.ReLU())
        self.done_head = nn.Linear(GW, 1)
        self.value_head = nn.Linear(GW, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        vox = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype)
        zc = zc.reshape(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        x = torch.cat([vox, zc], dim=1)
        flat = obs[:, 4 * Z:4 * Z + 3]
        h = F.relu(self.in3(x) + self.in2(flat).unsqueeze(2))
        h = F.relu(self.c2(h))
        e = self.cell(h).reshape(B, self.n_types, EMB, Z, Y, X)
        parts = []
        for t, (dy, dx) in enumerate(self.fps):
            et = e[:, t]
            mx = F.max_pool3d(et, (1, dy, dx), stride=1)
            mn = -F.max_pool3d(-et, (1, dy, dx), stride=1)
            av = F.avg_pool3d(et, (1, dy, dx), stride=1)
            p = torch.cat([mx, mn, av], dim=1)
            p = F.pad(p, (0, dx - 1, 0, dy - 1))
            parts.append(p)
        hp = torch.cat(parts, dim=1)
        hp = F.relu(self.mix(hp))
        place = self.out(hp)
        place = torch.logsumexp(place, dim=2).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3, 4)), h.amax((2, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
