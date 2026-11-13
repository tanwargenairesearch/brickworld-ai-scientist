import torch
import torch.nn as nn
import torch.nn.functional as F

C3 = 16
WIDTH = 64
HID = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = int(spec.Z)
        self.Z = Z
        self.c3a = nn.Conv3d(8, C3, 3, padding=1)
        self.c3b = nn.Conv3d(C3, C3, 3, padding=1)
        self.fuse = nn.Conv2d(C3 * Z, WIDTH, 1)
        if Z > 1:
            zc = torch.linspace(0.0, 1.0, Z)
        else:
            zc = torch.zeros(1)
        self.register_buffer('zpos', zc.view(1, 1, Z, 1, 1))
        self.bcast = nn.Linear(2 * WIDTH, WIDTH)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.cell = nn.Conv2d(WIDTH, HID, 1)
        self.type_mix = nn.ModuleList([nn.Conv2d(3 * HID, HID, 1) for _ in self.fps])
        self.type_out = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        vox = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        planes = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zp = self.zpos.expand(B, 1, Z, Y, X)
        x = torch.cat([vox, planes, zp], dim=1)
        h3 = F.relu(self.c3a(x))
        h3 = F.relu(self.c3b(h3))
        f = F.relu(self.fuse(h3.reshape(B, C3 * Z, Y, X)))
        g0 = self.bcast(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        f = F.relu(f + g0[:, :, None, None])
        h = self.cell(f)
        outs = []
        for (dy, dx), mix, out in zip(self.fps, self.type_mix, self.type_out):
            hp = F.pad(h, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(hp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-hp, (dy, dx), stride=1)
            av = F.avg_pool2d(hp, (dy, dx), stride=1)
            z = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            outs.append(out(z).flatten(1))
        place = torch.cat(outs, dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
