import math
import torch
import torch.nn as nn
import torch.nn.functional as F

C1 = 20
C2 = 32
W = 48


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        g2 = math.sqrt(2)
        self.l1 = ortho(nn.Conv2d(16, C1, 3, padding=1), g2)
        self.l2 = ortho(nn.Conv2d(C1, C2, 1), g2)
        self.gate = ortho(nn.Conv2d(C2, C2, 1), 1.0)
        self.c3 = ortho(nn.Conv2d(2 * C2, W, 3, padding=1), g2)
        self.place_head = ortho(nn.Conv2d(W, spec.n_types, 1), 0.01)
        self.glob = ortho(nn.Linear(2 * W, 64), g2)
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
        a = self.gate(h).reshape(B, Z, C2, Y, X)
        h = h.reshape(B, Z, C2, Y, X)
        w = torch.softmax(a, dim=1)
        att = (w * h).sum(1)
        f = torch.cat([att, h.amax(1)], dim=1)
        f = F.relu(self.c3(f))
        place = self.place_head(f).flatten(1)
        g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
