import torch
import torch.nn as nn
import torch.nn.functional as F

C3 = 24
WIDTH = 64
HID = 32


class FootHead(nn.Module):
    def __init__(self, c, footprints, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for (a, b) in footprints]
        self.pre = nn.Conv2d(c, hid, 1)
        self.mlps = nn.ModuleList()
        for _ in self.fps:
            out = nn.Conv2d(hid, 1, 1)
            nn.init.normal_(out.weight, std=0.01)
            nn.init.zeros_(out.bias)
            self.mlps.append(nn.Sequential(nn.Conv2d(3 * hid, hid, 1), nn.ReLU(), out))

    def forward(self, f):
        h = self.pre(f)
        outs = []
        for fp, mlp in zip(self.fps, self.mlps):
            dy, dx = fp
            hp = F.pad(h, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(hp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-hp, (dy, dx), stride=1)
            av = F.avg_pool2d(hp, (dy, dx), stride=1)
            outs.append(mlp(torch.cat([mx, mn, av], 1)))
        return torch.cat(outs, 1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.c1 = nn.Conv3d(8, C3, 3, padding=1)
        self.c2 = nn.Conv3d(C3, C3, (1, 3, 3), padding=(0, 1, 1))
        self.mix = nn.Sequential(nn.Conv2d(2 * C3, WIDTH, 1), nn.ReLU())
        self.place_head = FootHead(WIDTH, spec.footprints, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        Z = self.Z
        b, Y, X = obs.shape[0], obs.shape[2], obs.shape[3]
        per = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], dim=1)
        g2 = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(b, 3, Z, Y, X)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, 1, Z, 1, 1).expand(b, 1, Z, Y, X)
        x = torch.cat([per, g2, zc], dim=1)
        h = F.relu(self.c1(x))
        h = F.relu(self.c2(h))
        f = self.mix(torch.cat([h.amax(2), h.mean(2)], dim=1))
        place = self.place_head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
