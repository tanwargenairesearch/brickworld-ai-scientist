import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
C3 = 16
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.conv3 = nn.Conv3d(8, C3, 3, padding=1)
        self.mix = nn.Conv2d(C3 * self.Z, WIDTH, 1)
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(2)])
        self.cell = nn.Conv2d(WIDTH, HID, 1)
        self.type_heads = nn.ModuleList([nn.Conv2d(WIDTH + 2 * HID, 1, 1) for _ in self.footprints])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)
        for h in list(self.type_heads) + [self.done_head]:
            nn.init.orthogonal_(h.weight, gain=0.01)
            nn.init.zeros_(h.bias)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        vol = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        planes = obs[:, 4 * Z:4 * Z + 3].reshape(B, 3, 1, Y, X).expand(B, 3, Z, Y, X)
        zc = torch.arange(Z, dtype=obs.dtype, device=obs.device) / max(Z - 1, 1)
        zc = zc.reshape(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        v = torch.cat([vol, planes, zc], dim=1)
        h = torch.relu(self.conv3(v)).reshape(B, C3 * Z, Y, X)
        f = torch.relu(self.mix(h))
        for conv in self.blocks:
            f = f + torch.relu(conv(f))
        c = self.cell(f)
        outs = []
        for t, (dy, dx) in enumerate(self.footprints):
            cp = F.pad(c, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(cp, (dy, dx), stride=1)
            av = F.avg_pool2d(cp, (dy, dx), stride=1)
            outs.append(self.type_heads[t](torch.cat([f, mx, av], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
