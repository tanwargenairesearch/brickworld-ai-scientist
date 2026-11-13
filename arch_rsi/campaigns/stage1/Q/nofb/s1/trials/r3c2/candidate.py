import torch
import torch.nn as nn
import torch.nn.functional as F

W2 = 32
NBLOCK = 2
C3 = 16
HID = 12
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.fps = [(int(a), int(b)) for (a, b) in spec.footprints]
        self.inp = nn.Conv2d(spec.in_channels, W2, 3, padding=1)
        self.ctx = nn.Linear(2 * W2, W2)
        self.blocks = nn.ModuleList([nn.Conv2d(W2, W2, 3, padding=1) for _ in range(NBLOCK)])
        self.c1 = nn.Conv3d(8, C3, 3, padding=1)
        self.pre = nn.Conv3d(C3 + W2, HID, 1)
        self.heads = nn.ModuleList()
        for _ in self.fps:
            out = nn.Conv3d(HID, 1, 1)
            nn.init.normal_(out.weight, std=0.01)
            nn.init.zeros_(out.bias)
            self.heads.append(nn.Sequential(nn.Conv3d(3 * HID, HID, 1), nn.ReLU(), out))
        self.glob = nn.Sequential(nn.Linear(2 * W2 + 2 * C3, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        Z = self.Z
        b, Y, X = obs.shape[0], obs.shape[2], obs.shape[3]
        f = F.relu(self.inp(obs))
        c = F.relu(self.ctx(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)))
        f = f + c.unsqueeze(2).unsqueeze(3)
        for conv in self.blocks:
            f = f + F.relu(conv(f))
        per = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], dim=1)
        g2 = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(b, 3, Z, Y, X)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, 1, Z, 1, 1).expand(b, 1, Z, Y, X)
        x = torch.cat([per, g2, zc], dim=1)
        h = F.relu(self.c1(x))
        fz = f.unsqueeze(2).expand(b, W2, Z, Y, X)
        p = self.pre(torch.cat([h, fz], dim=1))
        outs = []
        for fp, head in zip(self.fps, self.heads):
            dy, dx = fp
            hp = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool3d(hp, (1, dy, dx), stride=1)
            mn = -F.max_pool3d(-hp, (1, dy, dx), stride=1)
            av = F.avg_pool3d(hp, (1, dy, dx), stride=1)
            s = head(torch.cat([mx, mn, av], 1)).squeeze(1)
            outs.append(torch.logsumexp(s, dim=1))
        place = torch.stack(outs, 1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), h.mean((2, 3, 4)), h.amax((2, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
