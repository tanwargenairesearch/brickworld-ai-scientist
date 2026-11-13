import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 16
GW = 64
HH = 8


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.T = spec.n_types
        self.fps = [(int(a), int(b)) for (a, b) in spec.footprints]
        self.in3 = nn.Conv3d(5, WIDTH, 3, padding=1)
        self.in2 = nn.Conv2d(3, WIDTH, 1)
        self.c2 = nn.Conv3d(WIDTH, WIDTH, 3, padding=1)
        self.heads = nn.ModuleList([nn.Conv3d(WIDTH, HH, (3, dy, dx)) for (dy, dx) in self.fps])
        self.hout = nn.Conv3d(self.T * HH, 2 * self.T, 1, groups=self.T)
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
        hs = []
        for t, (dy, dx) in enumerate(self.fps):
            p = F.pad(h, (0, dx - 1, 0, dy - 1, 1, 1))
            hs.append(self.heads[t](p))
        hh = F.relu(torch.cat(hs, dim=1))
        o = self.hout(hh).reshape(B, self.T, 2, Z, Y, X)
        q = o[:, :, 0]
        w = torch.softmax(o[:, :, 1], dim=2)
        place = (w * q).sum(2).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3, 4)), h.amax((2, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
