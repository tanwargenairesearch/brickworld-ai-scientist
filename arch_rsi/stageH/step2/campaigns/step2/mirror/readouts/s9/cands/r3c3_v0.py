import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 3}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
LD = 12
LHID = 16


def foot_pool(p, dy, dx):
    if dy * dx == 1:
        return torch.cat([p, p, p], dim=1)
    q = F.pad(p, (dx - 1, 0, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return torch.cat([mx, mn, av], dim=1)


class FlipPoolHead(nn.Module):
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
            h = F.relu(mix(foot_pool(p, dy, dx)))
            res.append(out(h))
        o = torch.cat(res, dim=1)
        return torch.flip(o, dims=[3])


class LayerHead(nn.Module):
    def __init__(self, spec, fdim, d, hid):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.d = d
        self.inp = nn.Conv2d(4 + self.Z + 3, d, 1)
        self.fproj = nn.Conv2d(fdim, d, 1)
        self.c2 = nn.Conv2d(d, d, 1)
        self.vmix = nn.ModuleList([nn.Conv2d(3 * d, hid, 1) for _ in self.fps])
        self.vout = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])
        self.gate = nn.ModuleList([nn.Conv2d(6 * d, 1, 1) for _ in self.fps])

    def forward(self, obs, f):
        B, _, Y, X = obs.shape
        Z = self.Z
        occ = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 2, 1, 3, 4)
        col = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        x = torch.cat([occ, eye, col], dim=2).reshape(B * Z, 7 + Z, Y, X)
        h = self.inp(x).view(B, Z, self.d, Y, X) + self.fproj(f).unsqueeze(1)
        h = F.relu(h).reshape(B * Z, self.d, Y, X)
        h = F.relu(self.c2(h))
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            t = foot_pool(h, dy, dx)
            v = self.vout[i](F.relu(self.vmix[i](t))).view(B, Z, Y, X)
            t5 = t.view(B, Z, 3 * self.d, Y, X)
            below = torch.cat([torch.zeros_like(t5[:, :1]), t5[:, :-1]], dim=1)
            g = self.gate[i](torch.cat([t5, below], dim=2).reshape(B * Z, 6 * self.d, Y, X)).view(B, Z, Y, X)
            w = torch.softmax(g, dim=1)
            res.append((w * v).sum(1))
        o = torch.stack(res, dim=1)
        return torch.flip(o, dims=[3])


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = FlipPoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.lhead = LayerHead(spec, WIDTH, LD, LHID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = (self.head(f) + self.lhead(obs, f)).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
