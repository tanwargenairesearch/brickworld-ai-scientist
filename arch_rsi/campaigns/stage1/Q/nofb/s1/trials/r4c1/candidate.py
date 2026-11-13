import torch
import torch.nn as nn
import torch.nn.functional as F

C3 = 24
HID = 12
G = 64


def volume(obs, Z):
    b, Y, X = obs.shape[0], obs.shape[2], obs.shape[3]
    per = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], dim=1)
    g2 = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(b, 3, Z, Y, X)
    zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, 1, Z, 1, 1).expand(b, 1, Z, Y, X)
    return torch.cat([per, g2, zc], dim=1)


class GatedFootHead3D(nn.Module):
    def __init__(self, c, footprints, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for (a, b) in footprints]
        self.pre = nn.Conv3d(c, hid, 1)
        self.heads = nn.ModuleList()
        for _ in self.fps:
            out = nn.Conv3d(hid, 2, 1)
            nn.init.normal_(out.weight, std=0.01)
            nn.init.zeros_(out.bias)
            self.heads.append(nn.Sequential(nn.Conv3d(3 * hid, hid, 1), nn.ReLU(), out))

    def forward(self, h):
        p = self.pre(h)
        outs = []
        for fp, head in zip(self.fps, self.heads):
            dy, dx = fp
            hp = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool3d(hp, (1, dy, dx), stride=1)
            mn = -F.max_pool3d(-hp, (1, dy, dx), stride=1)
            av = F.avg_pool3d(hp, (1, dy, dx), stride=1)
            s = head(torch.cat([mx, mn, av], 1))
            w = torch.softmax(s[:, 1], dim=1)
            outs.append((w * s[:, 0]).sum(1))
        return torch.stack(outs, 1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.c1 = nn.Conv3d(8, C3, 3, padding=1)
        self.ctx = nn.Linear(2 * C3, C3)
        self.zpre = nn.Conv3d(C3, HID, 1)
        self.zmix = nn.Conv3d(C3 + 2 * HID, C3, 1)
        self.c2 = nn.Conv3d(C3, C3, (1, 3, 3), padding=(0, 1, 1))
        self.head = GatedFootHead3D(C3, spec.footprints, HID)
        self.glob = nn.Sequential(nn.Linear(2 * C3, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        b = obs.shape[0]
        x = volume(obs, self.Z)
        h = F.relu(self.c1(x))
        c = F.relu(self.ctx(torch.cat([h.mean((2, 3, 4)), h.amax((2, 3, 4))], dim=1)))
        h = h + c.view(b, -1, 1, 1, 1)
        q = self.zpre(h)
        below = torch.cummax(q, dim=2).values
        above = torch.flip(torch.cummax(torch.flip(q, [2]), dim=2).values, [2])
        h = h + F.relu(self.zmix(torch.cat([h, below, above], dim=1)))
        h = h + F.relu(self.c2(h))
        place = self.head(h)
        g = self.glob(torch.cat([h.mean((2, 3, 4)), h.amax((2, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
