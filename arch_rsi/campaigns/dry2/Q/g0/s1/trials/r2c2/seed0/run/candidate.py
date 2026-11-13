import torch
import torch.nn as nn
import torch.nn.functional as F

GAIN = 10.0
WIDTH = 32
HID = 48
NF = 6
NG = 6


def shift(t, oy, ox):
    if oy == 0 and ox == 0:
        return t
    Y, X = t.shape[-2], t.shape[-1]
    p = F.pad(t, (0, ox, 0, oy))
    return p[..., oy:oy + Y, ox:ox + X]


def type_features(obs, spec):
    Z = spec.Z
    missing = obs[:, 2 * Z:3 * Z]
    hb = torch.round(obs[:, 4 * Z:4 * Z + 1] * Z)
    zr = torch.arange(Z, device=obs.device, dtype=obs.dtype).view(1, Z, 1, 1)
    out = []
    for (dy, dx) in spec.footprints:
        offs = [(oy, ox) for oy in range(dy) for ox in range(dx)]
        hs = [shift(hb, oy, ox) for (oy, ox) in offs]
        hmax = hs[0]
        hmin = hs[0]
        for h in hs[1:]:
            hmax = torch.maximum(hmax, h)
            hmin = torch.minimum(hmin, h)
        at = (zr == hmax).to(obs.dtype)
        fill = torch.zeros_like(hmax)
        buried = torch.zeros_like(hmax)
        for (oy, ox), h in zip(offs, hs):
            ms = shift(missing, oy, ox)
            fill = fill + (ms * at).sum(1, keepdim=True)
            gap = ((zr >= h) & (zr < hmax)).to(obs.dtype)
            buried = buried + (ms * gap).sum(1, keepdim=True)
        n = float(dy * dx)
        f = torch.cat([fill / n, (n - fill) / n, buried / n, hmax / Z,
                       (hmax - hmin) / Z, torch.full_like(hmax, n / 4.0)], 1)
        out.append(f)
    return out


def global_features(obs, spec, feats):
    Z = spec.Z
    tsum = obs[:, Z:2 * Z].sum((1, 2, 3)).clamp(min=1.0)
    miss = obs[:, 2 * Z:3 * Z].sum((1, 2, 3)) / tsum
    extra = obs[:, 3 * Z:4 * Z].sum((1, 2, 3)) / tsum
    step = obs[:, 4 * Z + 2].mean((1, 2))
    best = torch.stack([(f[:, 0] - f[:, 2]).amax((1, 2)) for f in feats], 1).amax(1)
    anyfill = torch.stack([f[:, 0].amax((1, 2)) for f in feats], 1).amax(1)
    done_flag = (miss < 1e-6).to(obs.dtype)
    return torch.stack([miss, extra, step, best, anyfill, done_flag], 1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        nt = spec.n_types
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        cin = NF + nt
        self.lin = nn.Conv2d(cin, 1, 1)
        self.mlp = nn.Sequential(nn.Conv2d(cin + WIDTH, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        self.done_head = nn.Linear(NG, 1)
        nn.init.zeros_(self.done_head.weight)
        nn.init.zeros_(self.done_head.bias)
        self.register_buffer('type_eye', torch.eye(nt))
        self.value = nn.Sequential(nn.Linear(NG + 2 * NF * nt + 2 * WIDTH, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = obs.shape[2], obs.shape[3]
        nt = self.spec.n_types
        feats = type_features(obs, self.spec)
        f = self.body(obs)
        xs = []
        for t in range(nt):
            oh = self.type_eye[t].view(1, nt, 1, 1).expand(B, nt, Y, X)
            xs.append(torch.cat([feats[t], oh], 1))
        x = torch.stack(xs, 0).reshape(nt * B, NF + nt, Y, X)
        fc = f.unsqueeze(0).expand(nt, B, WIDTH, Y, X).reshape(nt * B, WIDTH, Y, X)
        s = GAIN * (self.lin(x) + self.mlp(torch.cat([x, fc], 1)))
        s = s.reshape(nt, B, Y * X).transpose(0, 1).reshape(B, nt * Y * X)
        gf = global_features(obs, self.spec, feats)
        done = GAIN * self.done_head(gf)
        pooled = torch.cat([torch.cat([ft.mean((2, 3)), ft.amax((2, 3))], 1) for ft in feats], 1)
        v = self.value(torch.cat([gf, pooled, f.mean((2, 3)), f.amax((2, 3))], 1)).squeeze(1)
        logits = torch.cat([s, done], 1)
        return logits, v


def build_policy(spec):
    return Net(spec)
