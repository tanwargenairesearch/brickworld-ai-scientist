import torch
import torch.nn as nn
import torch.nn.functional as F

W2 = 48
C3 = 12


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, W2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(W2, W2, 3, padding=1), nn.ReLU())
        self.v1 = nn.Conv3d(5, C3, 3, padding=1)
        self.f2v = nn.Conv2d(W2, C3, 1)
        self.v2 = nn.Conv3d(C3, C3, 3, padding=1)
        self.score = nn.ModuleList([nn.Conv3d(C3, 1, (1, dy, dx)) for dy, dx in self.fps])
        self.land = nn.ModuleList([nn.Conv2d(W2, spec.Z, (dy, dx)) for dy, dx in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W2 + 2 * C3, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        Z = self.Z
        B, _, Y, X = obs.shape
        f = self.body(obs)
        vox = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], dim=1)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        vox = torch.cat([vox, zc], dim=1)
        v = F.relu(self.v1(vox) + self.f2v(f).unsqueeze(2))
        v = F.relu(v + self.v2(v))
        outs = []
        for (dy, dx), sc, ld in zip(self.fps, self.score, self.land):
            vp = F.pad(v, (0, dx - 1, 0, dy - 1))
            s = sc(vp).squeeze(1)
            fp = F.pad(f, (0, dx - 1, 0, dy - 1))
            w = torch.softmax(ld(fp), dim=1)
            outs.append((w * s).sum(1, keepdim=True))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), v.mean((2, 3, 4)), v.amax((2, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
