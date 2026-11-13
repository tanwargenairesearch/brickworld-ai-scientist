import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 24
HID = 16
GW = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        c_in = 4 + 3 + 1
        self.c1 = nn.Conv3d(c_in, WIDTH, 3, padding=1)
        self.c2 = nn.Conv3d(WIDTH, WIDTH, 3, padding=1)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv3d(2 * WIDTH, HID, 1), nn.ReLU(), nn.Conv3d(HID, 1, 1))
            for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GW), nn.ReLU())
        self.done_head = nn.Linear(GW, 1)
        self.value_head = nn.Linear(GW, 1)
        if self.Z > 1:
            zc = torch.linspace(0.0, 1.0, self.Z)
        else:
            zc = torch.zeros(1)
        self.register_buffer('zc', zc)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        vox = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        planes = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = self.zc.view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        x = torch.cat([vox, planes, zc], dim=1)
        f = F.relu(self.c1(x))
        f = f + F.relu(self.c2(f))
        outs = []
        for fp, head in zip(self.fps, self.heads):
            dy, dx = fp
            if dy == 1 and dx == 1:
                mx = f
                mn = f
            else:
                p = F.pad(f, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool3d(p, (1, dy, dx), stride=1)
                mn = -F.max_pool3d(-p, (1, dy, dx), stride=1)
            s = head(torch.cat([mx, mn], dim=1))
            outs.append(s.amax(dim=2))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3, 4)), f.amax((2, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
