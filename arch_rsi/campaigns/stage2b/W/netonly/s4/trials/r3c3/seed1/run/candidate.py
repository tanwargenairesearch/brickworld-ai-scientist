import math
import torch
import torch.nn as nn
import torch.nn.functional as F

C1 = 36
C2 = 48


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.fps = [(int(f[0]), int(f[1])) for f in spec.footprints]
        g2 = math.sqrt(2)
        self.l1 = ortho(nn.Conv2d(16, C1, 3, padding=1), g2)
        self.l2 = ortho(nn.Conv2d(C1, C2, 1), g2)
        P = 2 * C2
        self.fp_score = nn.ModuleList([ortho(nn.Conv2d(P, 1, 1), 0.01) for _ in self.fps])
        self.fp_gate = nn.ModuleList([ortho(nn.Conv2d(P, 1, 1), 1.0) for _ in self.fps])
        self.glob = ortho(nn.Linear(2 * C2, 64), g2)
        self.done_head = ortho(nn.Linear(64, 1), 0.01)
        self.value_head = ortho(nn.Linear(64, 1), 1.0)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        s = F.pad(s, (0, 0, 0, 0, 1, 1))
        loc = torch.cat([s[:, :, 0:Z], s[:, :, 1:Z + 1], s[:, :, 2:Z + 2]], dim=1)
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        x = torch.cat([loc, gl, zc], dim=1)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * Z, 16, Y, X)
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        hz = h.reshape(B, Z, C2, Y, X)
        outs = []
        for t, (dy, dx) in enumerate(self.fps):
            hp = F.pad(h, (0, dx - 1, 0, dy - 1))
            m = F.max_pool2d(hp, (dy, dx), stride=1)
            a = F.avg_pool2d(hp, (dy, dx), stride=1)
            p = torch.cat([m, a], dim=1)
            sc = self.fp_score[t](p).reshape(B, Z, Y, X)
            gt = self.fp_gate[t](p).reshape(B, Z, Y, X)
            w = torch.softmax(gt, dim=1)
            outs.append((w * sc).sum(1))
        place = torch.stack(outs, dim=1).flatten(1)
        pooled = torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1)
        g = F.relu(self.glob(pooled))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
