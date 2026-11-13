import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
HID = 64
NF = 5


def type_features(obs, spec):
    Z = spec.Z
    target = obs[:, Z:2 * Z]
    missing = obs[:, 2 * Z:3 * Z]
    hb = torch.round(obs[:, 4 * Z:4 * Z + 1] * Z)
    zr = torch.arange(Z, device=obs.device, dtype=obs.dtype).view(1, Z, 1, 1)
    out = []
    for (dy, dx) in spec.footprints:
        pad = (0, dx - 1, 0, dy - 1)
        hmax = F.max_pool2d(F.pad(hb, pad), (dy, dx), stride=1)
        hmin = -F.max_pool2d(F.pad(-hb, pad), (dy, dx), stride=1)
        ms = F.avg_pool2d(F.pad(missing, pad), (dy, dx), stride=1)
        ts = F.avg_pool2d(F.pad(target, pad), (dy, dx), stride=1)
        at = (zr == hmax).to(obs.dtype)
        below = (zr < hmax).to(obs.dtype)
        m_at = (ms * at).sum(1, keepdim=True)
        t_at = (ts * at).sum(1, keepdim=True)
        m_below = (ms * below).sum(1, keepdim=True)
        out.append(torch.cat([m_at, 1.0 - t_at, m_below, hmax / Z, (hmax - hmin) / Z], 1))
    return out


def global_features(obs, spec, feats):
    Z = spec.Z
    tsum = obs[:, Z:2 * Z].sum((1, 2, 3)).clamp(min=1.0)
    miss = obs[:, 2 * Z:3 * Z].sum((1, 2, 3)) / tsum
    extra = obs[:, 3 * Z:4 * Z].sum((1, 2, 3)) / tsum
    step = obs[:, 4 * Z + 2].mean((1, 2))
    best = torch.stack([f[:, 0].amax((1, 2)) for f in feats], 1).amax(1)
    return torch.stack([miss, extra, step, best], 1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(
            nn.Conv2d(WIDTH + NF * spec.n_types, HID, 1), nn.ReLU(),
            nn.Conv2d(HID, spec.n_types, 1))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + 4, HID), nn.ReLU())
        self.done_head = nn.Linear(HID, 1)
        self.value_head = nn.Linear(HID, 1)

    def forward(self, obs, mask):
        feats = type_features(obs, self.spec)
        f = self.body(obs)
        place = self.head(torch.cat([f] + feats, 1)).flatten(1)
        gf = global_features(obs, self.spec, feats)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), gf], 1))
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
