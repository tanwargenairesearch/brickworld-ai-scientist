import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
LCTX = 24
LH = 48
LP = 16
LHID = 16

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.99, "gae_lambda": 0.95}


def foot_pool(p, dy, dx):
    if dy * dx == 1:
        return torch.cat([p, p, p], dim=1)
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return torch.cat([mx, mn, av], dim=1)


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
            h = F.relu(mix(foot_pool(p, dy, dx)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class LayerHead(nn.Module):
    def __init__(self, spec, cin):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.ctx = nn.Conv2d(cin, LCTX, 1)
        self.l1 = nn.Conv2d(8 + LCTX + self.Z, LH, 1)
        self.proj = nn.Conv2d(LH, LP, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * LP, LHID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(LHID, 1, 1) for _ in self.fps])

    def forward(self, obs, f):
        B, _, Y, X = obs.shape
        Z = self.Z
        lay = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        below = F.pad(lay, (0, 0, 0, 0, 1, 0))[:, :, :Z]
        lay = torch.cat([lay, below], dim=1).permute(0, 2, 1, 3, 4)
        ctx = self.ctx(f).unsqueeze(1).expand(B, Z, LCTX, Y, X)
        zp = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        h = torch.cat([lay, ctx, zp], dim=2).reshape(B * Z, 8 + LCTX + Z, Y, X)
        h = F.relu(self.l1(h))
        p = self.proj(h)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            res.append(out(F.relu(mix(foot_pool(p, dy, dx)))))
        s = torch.cat(res, dim=1).reshape(B, Z, -1)
        return torch.logsumexp(s, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.lhead = LayerHead(spec, WIDTH)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f) + self.lhead(obs, f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
