import torch
import torch.nn as nn
import torch.nn.functional as F

TR = 32
LH = 32
LP = 12
TP = 16
HID = 32


def pool3(p, dy, dx):
    if dy * dx == 1:
        return torch.cat([p, p, p], dim=1)
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return torch.cat([mx, mn, av], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        C = int(spec.in_channels)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Conv2d(C, TR, 1)
        self.c1 = nn.Conv2d(TR, TR, 3, padding=1)
        self.c2 = nn.Conv2d(TR, TR, 3, padding=1)
        self.col_lin = nn.Conv2d(TR, LH, 1)
        self.z_lin = nn.Conv2d(12 + Z, LH, 1, bias=False)
        self.l_out = nn.Conv2d(LH, LP, 1)
        self.tproj = nn.Conv2d(TR, TP, 1)
        self.mix_l = nn.ModuleList([nn.Conv2d(3 * LP, HID, 1) for _ in self.fps])
        self.mix_t = nn.ModuleList([nn.Conv2d(3 * TP, HID, 1, bias=False) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * TR, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = F.relu(self.stem(obs))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))

        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        own = torch.stack([b, t, m, e], dim=2)
        zero = torch.zeros_like(own[:, :1])
        below = torch.cat([zero, own[:, :-1]], dim=1)
        above = torch.cat([own[:, 1:], zero], dim=1)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        zin = torch.cat([own, below, above, eye], dim=2).reshape(B * Z, 12 + Z, Y, X)
        zp = self.z_lin(zin).view(B, Z, LH, Y, X)
        cp = self.col_lin(f).unsqueeze(1)
        h = F.relu(zp + cp).reshape(B * Z, LH, Y, X)
        q = F.relu(self.l_out(h))
        tp = self.tproj(f)

        res = []
        for (dy, dx), ml, mt, out in zip(self.fps, self.mix_l, self.mix_t, self.outs):
            lq = ml(pool3(q, dy, dx)).view(B, Z, HID, Y, X)
            tq = mt(pool3(tp, dy, dx)).unsqueeze(1)
            hh = F.relu(lq + tq).reshape(B * Z, HID, Y, X)
            sc = out(hh).view(B, Z, Y, X)
            res.append(torch.logsumexp(sc, dim=1))
        place = torch.stack(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
