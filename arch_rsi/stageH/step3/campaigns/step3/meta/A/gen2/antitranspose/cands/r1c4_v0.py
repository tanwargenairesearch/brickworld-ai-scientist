import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
SKIP = 16
PROJ = 16
HID = 16
WHID = 32
GHID = 64


def ortho_init(module, gain):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def fp_pool(p, dy, dx, pad):
    if dy * dx == 1:
        return torch.cat([p, p, p], dim=1)
    q = F.pad(p, pad)
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return torch.cat([mx, mn, av], dim=1)


def corner_pad(dy, dx, c):
    if c == 0:
        return (0, dx - 1, 0, dy - 1)
    if c == 1:
        return (dx - 1, 0, 0, dy - 1)
    if c == 2:
        return (0, dx - 1, dy - 1, 0)
    return (dx - 1, 0, dy - 1, 0)


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = nn.Conv2d(cin, SKIP, 1)
        self.out_ch = WIDTH + SKIP

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        k = F.relu(self.skip(obs))
        return torch.cat([x, k], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        self.hyps = [(c, s) for s in range(2) for c in range(4)]
        nh = len(self.hyps)
        self.trunk = Trunk(spec.in_channels)
        c = self.trunk.out_ch
        self.glob = nn.Sequential(nn.Linear(2 * c, GHID), nn.ReLU())
        self.done_head = nn.Linear(GHID, 1)
        self.value_head = nn.Linear(GHID, 1)
        self.type_bias = nn.Linear(GHID, T)
        self.proj = nn.Conv2d(c, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ * T, HID * T, 1, groups=T) for _ in range(nh)])
        self.outs = nn.ModuleList([nn.Conv2d(HID * T, T, 1, groups=T) for _ in range(nh)])
        self.win = nn.Conv2d(PROJ, WHID, 3, padding=1)
        self.win_out = nn.Conv2d(WHID, T, 1)
        self.sel = nn.Parameter(torch.zeros(nh + 1))
        ortho_init(self, math.sqrt(2))
        ortho_init(self.done_head, 0.01)
        ortho_init(self.type_bias, 0.01)
        ortho_init(self.outs, 0.01)
        ortho_init(self.win_out, 0.01)
        ortho_init(self.value_head, 1.0)

    def forward(self, obs, mask):
        f = self.trunk(obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        tb = self.type_bias(g)[:, :, None, None]
        p = self.proj(f)
        outs = []
        for i, (cn, sw) in enumerate(self.hyps):
            zs = []
            for dy, dx in self.fps:
                a, b = (dx, dy) if sw == 1 else (dy, dx)
                zs.append(fp_pool(p, a, b, corner_pad(a, b, cn)))
            z = torch.cat(zs, dim=1)
            l = self.outs[i](F.relu(self.mix[i](z))) + tb
            outs.append(l.flatten(1))
        lw = self.win_out(F.relu(self.win(p))) + tb
        outs.append(lw.flatten(1))
        w = F.softmax(5.0 * self.sel, dim=0)
        place = (torch.stack(outs, 1) * w[None, :, None]).sum(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
