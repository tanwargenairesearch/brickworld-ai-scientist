import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "n_minibatches": 16, "ppo_epochs": 1, "gamma": 0.97, "gae_lambda": 0.92}

TR = 32
LH = 32
LP = 8
TP = 16
HID = 40


def pool_fp(p, dy, dx):
    if dy * dx == 1:
        return p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    return mx, mn


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
        self.c3 = nn.Conv2d(TR, TR, 3, padding=2, dilation=2)
        self.tproj = nn.Conv2d(TR, TP, 1)
        self.col_lin = nn.Conv2d(C, LH, 1)
        self.z_lin = nn.Conv2d(8 + Z, LH, 1, bias=False)
        self.l_out = nn.Conv2d(LH, LP, 1)
        nin = Z * 2 * LP + 3 * TP
        self.mix = nn.ModuleList([nn.Conv2d(nin, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * TR, 64), nn.ReLU())
        self.gctx = nn.Linear(64, HID, bias=False)
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = F.relu(self.stem(obs))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        f = F.relu(f + self.c3(f))
        tp = self.tproj(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        gc = self.gctx(g).view(B, HID, 1, 1)

        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        own = torch.stack([b, t, m, e], dim=2)
        below = torch.cat([torch.zeros_like(own[:, :1]), own[:, :-1]], dim=1)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        zin = torch.cat([own, below, eye], dim=2).reshape(B * Z, 8 + Z, Y, X)
        zp = self.z_lin(zin).view(B, Z, LH, Y, X)
        cp = self.col_lin(obs).unsqueeze(1)
        h = F.relu(zp + cp).reshape(B * Z, LH, Y, X)
        q = F.relu(self.l_out(h))

        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            lmx, lmn = pool_fp(q, dy, dx)
            lf = torch.cat([lmx, lmn], dim=1).reshape(B, Z * 2 * LP, Y, X)
            if dy * dx == 1:
                tmx, tmn, tav = tp, tp, tp
            else:
                pp = F.pad(tp, (0, dx - 1, 0, dy - 1))
                tmx = F.max_pool2d(pp, (dy, dx), stride=1)
                tmn = -F.max_pool2d(-pp, (dy, dx), stride=1)
                tav = F.avg_pool2d(pp, (dy, dx), stride=1)
            hh = F.relu(mix(torch.cat([lf, tmx, tmn, tav], dim=1)) + gc)
            res.append(out(hh))
        place = torch.cat(res, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
