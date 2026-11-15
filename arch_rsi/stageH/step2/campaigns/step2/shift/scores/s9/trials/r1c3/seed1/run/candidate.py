import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "lr": 0.001}

W = 16
PROJ = 12
HID = 20
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
        return torch.cat(res, dim=1)


class LayerNet(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.C = int(spec.in_channels)
        self.T = len(spec.footprints)
        self.register_buffer("zpos", torch.eye(self.Z).view(1, self.Z, self.Z, 1, 1))
        cin = 8 + self.C + self.Z
        self.stem = nn.Conv2d(cin, W, 1)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.colmix = nn.Conv2d(W, W, 1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.head = PoolHead(W, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        g = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        below = F.pad(g, (0, 0, 0, 0, 1, 0))[:, :, :Z]
        per = torch.cat([g, below], 1).permute(0, 2, 1, 3, 4)
        full = obs.unsqueeze(1).expand(B, Z, C, Y, X)
        zp = self.zpos.expand(B, Z, Z, Y, X)
        inp = torch.cat([per, full, zp], 2).reshape(B * Z, 8 + C + Z, Y, X)
        s = F.relu(self.stem(inp))
        h = F.relu(self.c1(s))
        col = h.reshape(B, Z, W, Y, X).mean(1)
        cm = self.colmix(col).unsqueeze(1).expand(B, Z, W, Y, X).reshape(B * Z, W, Y, X)
        h = F.relu(h + cm)
        f = F.relu(h + self.c2(h))
        place = self.head(f).reshape(B, Z, self.T, Y, X).sum(1).reshape(B, self.T * Y * X)
        fz = f.reshape(B, Z, W, Y, X)
        gf = self.glob(torch.cat([fz.mean((1, 3, 4)), fz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(gf)], dim=1)
        return logits, self.value_head(gf).squeeze(1)


def build_policy(spec):
    return LayerNet(spec)
