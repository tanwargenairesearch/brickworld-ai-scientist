import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
C3 = 16
HID = 16


def fp_max(x, dy, dx):
    return F.max_pool2d(F.pad(x, (0, dx - 1, 0, dy - 1)), (dy, dx), stride=1)


def fp_avg(x, dy, dx):
    return F.avg_pool2d(F.pad(x, (0, dx - 1, 0, dy - 1)), (dy, dx), stride=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.c3a = nn.Conv3d(8, C3, 3, padding=1)
        self.c3b = nn.Conv3d(C3, C3, 3, padding=1)
        self.mix = nn.Conv2d(C3 * self.Z, WIDTH, 1)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hcell = nn.Conv2d(WIDTH, HID, 1)
        self.land = nn.ModuleList([nn.Conv2d(HID, self.Z, 1) for _ in self.footprints])
        self.type_heads = nn.ModuleList([nn.Conv2d(WIDTH + 2 * C3, 1, 1) for _ in self.footprints])
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
        vin = torch.cat([vol, planes, zc], dim=1)
        v = torch.relu(self.c3a(vin))
        v = v + torch.relu(self.c3b(v))
        vf = v.reshape(B, C3 * Z, Y, X)
        f = torch.relu(self.mix(vf))
        f = f + torch.relu(self.conv2(f))
        hc = torch.relu(self.hcell(f))
        outs = []
        for t, (dy, dx) in enumerate(self.footprints):
            p = torch.softmax(self.land[t](fp_max(hc, dy, dx)), dim=1).unsqueeze(1)
            vmin = (-fp_max(-vf, dy, dx)).reshape(B, C3, Z, Y, X)
            vmean = fp_avg(vf, dy, dx).reshape(B, C3, Z, Y, X)
            smin = (vmin * p).sum(2)
            smean = (vmean * p).sum(2)
            outs.append(self.type_heads[t](torch.cat([f, smin, smean], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
