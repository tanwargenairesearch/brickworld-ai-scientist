import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'ppo_epochs': 2}

W = 24
PROJ = 16
HID = 16
GH = 64


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.X = int(spec.X)
        self.Z = int(spec.Z)
        c = spec.in_channels
        self.lin = nn.Conv2d(25, W, 1)
        self.glin = nn.Conv2d(2 * c, W, 1)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.head = PoolHead(W, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        r = torch.roll(obs, shifts=self.X // 2, dims=3)
        o2 = torch.cat([obs, r], dim=1)
        lay = torch.cat([obs[:, :4 * Z], r[:, :4 * Z]], dim=1).reshape(B, 8, Z, Y, X)
        lay = F.pad(lay, (0, 0, 0, 0, 1, 1))
        per = torch.cat([lay[:, :, 0:Z], lay[:, :, 1:Z + 1], lay[:, :, 2:Z + 2]], dim=1)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        per = torch.cat([per, zc], dim=1)
        per = per.permute(0, 2, 1, 3, 4).reshape(B * Z, 25, Y, X)
        g = self.glin(o2).unsqueeze(1).expand(B, Z, W, Y, X).reshape(B * Z, W, Y, X)
        h = F.relu(self.lin(per) + g)
        x = F.relu(self.c1(h))
        f = F.relu(x + self.c2(x))
        place = self.head(f).reshape(B, Z, -1)
        place = torch.logsumexp(place, dim=1)
        fz = f.reshape(B, Z, W, Y, X)
        gf = torch.cat([fz.mean((1, 3, 4)), fz.amax((1, 3, 4))], dim=1)
        gg = self.glob(gf)
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
