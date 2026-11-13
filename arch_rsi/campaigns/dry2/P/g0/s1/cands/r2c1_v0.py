import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 32
HID = 32
GAIN = 8.0
USE_COUNTS = True


def fp_pool(x, dy, dx, mode):
    if dy > 1 or dx > 1:
        x = F.pad(x, (0, dx - 1, 0, dy - 1))
    if mode == 'max':
        return F.max_pool2d(x, (dy, dx), stride=1)
    return F.avg_pool2d(x, (dy, dx), stride=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = spec.Z
        self.Z = Z
        self.T = spec.n_types
        self.fps = [(int(a), int(b)) for (a, b) in spec.footprints]
        C = spec.in_channels
        self.body = nn.Sequential(
            nn.Conv2d(C, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        nf = 11 if USE_COUNTS else 9
        self.lin = nn.Conv2d(nf + self.T, 1, 1)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        self.head1 = nn.Conv2d(WIDTH + nf + self.T, HID, 1)
        self.head2 = nn.Conv2d(HID, 1, 1)
        nn.init.uniform_(self.head2.weight, -0.01, 0.01)
        nn.init.zeros_(self.head2.bias)
        NG = 6
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + NG, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        self.done_lin = nn.Linear(NG, 1)
        nn.init.zeros_(self.done_lin.weight)
        nn.init.zeros_(self.done_lin.bias)
        self.value_lin = nn.Linear(NG, 1)
        nn.init.zeros_(self.value_lin.weight)
        nn.init.zeros_(self.value_lin.bias)

    def feats(self, obs, dy, dx, ntar):
        Z = self.Z
        hb = torch.round(obs[:, 4 * Z:4 * Z + 1] * Z)
        L = fp_pool(hb, dy, dx, 'max')
        zs = torch.arange(Z, device=obs.device, dtype=obs.dtype).view(1, -1, 1, 1)
        oh = (L == zs).to(obs.dtype)
        tavg = fp_pool(obs[:, Z:2 * Z], dy, dx, 'avg')
        mavg = fp_pool(obs[:, 2 * Z:3 * Z], dy, dx, 'avg')
        good = (oh * tavg).sum(1, keepdim=True)
        valid = (L < Z).to(obs.dtype)
        bad = (1.0 - good) * valid
        tA = (oh[:, :Z - 1] * tavg[:, 1:]).sum(1, keepdim=True)
        below = (L > zs).to(obs.dtype)
        gm = (below * mavg).sum(1, keepdim=True)
        bury = (gm > 1e-6).to(obs.dtype)
        full = (good > 0.999).to(obs.dtype)
        flat = (L - fp_pool(hb, dy, dx, 'avg')) / Z
        area = torch.ones_like(L) * (dy * dx / 4.0)
        fl = [good, bad, gm, bury, full, tA, flat, L / Z, area]
        if USE_COUNTS:
            fl = fl + [good * (dy * dx) / ntar, bad * (dy * dx) / ntar]
        return torch.cat(fl, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        f = self.body(obs)
        tar = obs[:, Z:2 * Z].sum((1, 2, 3)).clamp(min=1.0)
        ntar = tar.view(B, 1, 1, 1)
        fts, hs, tos = [], [], []
        for t, (dy, dx) in enumerate(self.fps):
            fts.append(fp_pool(f, dy, dx, 'avg'))
            hs.append(self.feats(obs, dy, dx, ntar))
            to = torch.zeros(B, self.T, Y, X, device=obs.device, dtype=obs.dtype)
            to[:, t] = 1.0
            tos.append(to)
        ft = torch.cat(fts, 0)
        h = torch.cat(hs, 0)
        to = torch.cat(tos, 0)
        hl = torch.cat([h, to], 1)
        p = self.lin(hl * GAIN) + self.head2(F.relu(self.head1(torch.cat([ft, hl], 1))))
        place = p.reshape(self.T, B, Y * X).permute(1, 0, 2).reshape(B, -1)
        miss = obs[:, 2 * Z:3 * Z].sum((1, 2, 3))
        ext = obs[:, 3 * Z:4 * Z].sum((1, 2, 3))
        gf = torch.stack([miss / tar, (miss > 0.5).to(obs.dtype), ext / tar,
                          (ext > 0.5).to(obs.dtype), obs[:, 4 * Z + 2].mean((1, 2)),
                          ((miss + ext) < 0.5).to(obs.dtype)], 1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), gf], 1))
        done = self.done_head(g) + self.done_lin(gf * GAIN)
        value = self.value_head(g) + self.value_lin(gf)
        logits = torch.cat([place, done], 1)
        return logits, value.squeeze(1)


def build_policy(spec):
    return Net(spec)
