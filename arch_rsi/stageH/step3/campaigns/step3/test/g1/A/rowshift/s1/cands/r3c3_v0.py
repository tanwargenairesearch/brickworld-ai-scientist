import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
W = 40
W2 = 48
SKIP = 16
DET = 16
PROJ = 32
HID = 32
GLOB = 64

RECIPE = {'ppo_epochs': 2, 'n_envs': 256, 'rollout_len': 32, 'n_minibatches': 32, 'lr': 0.0007, 'gamma': 0.995, 'gae_lambda': 0.97, 'clip_eps': 0.3, 'ent_coef': 0.05}


def init_layers(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, math.sqrt(2.0))
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def init_out(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)


def fp_pool(p, dy, dx):
    if dy * dx == 1:
        return torch.cat([p, p, p], dim=1)
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return torch.cat([mx, mn, av], dim=1)


def coords(obs):
    bsz = obs.shape[0]
    ny = obs.shape[2]
    nx = obs.shape[3]
    yy = torch.linspace(-1.0, 1.0, ny, device=obs.device, dtype=obs.dtype).view(1, 1, ny, 1).expand(bsz, 1, ny, nx)
    xx = torch.linspace(-1.0, 1.0, nx, device=obs.device, dtype=obs.dtype).view(1, 1, 1, nx).expand(bsz, 1, ny, nx)
    return torch.cat([yy, xx], dim=1)


def mask_map(mask, nt, ny, nx, dtype):
    return mask[:, :nt * ny * nx].to(dtype).reshape(mask.shape[0], nt, ny, nx)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        nt = len(self.fps)
        self.nt = nt
        cin = c + 2 + nt
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.GroupNorm(8, W), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.n1 = nn.GroupNorm(8, W)
        self.d1 = nn.Conv2d(W, W2, 3, padding=1)
        self.nd1 = nn.GroupNorm(8, W2)
        self.d2 = nn.Conv2d(W2, W2, 3, padding=1)
        self.nd2 = nn.GroupNorm(8, W2)
        self.u1 = nn.Conv2d(W2, W, 1)
        self.u2 = nn.Conv2d(W2, W, 1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.n2 = nn.GroupNorm(8, W)
        self.skip = nn.Conv2d(cin, SKIP, 1)
        self.proj = nn.Conv2d(W + SKIP, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.det = nn.Conv2d(cin, DET, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W + W2 + 2 * DET, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        init_layers(self)
        for o in self.outs:
            init_out(o, 0.01)
        init_out(self.done_head, 0.01)
        init_out(self.value_head, 1.0)

    def forward(self, obs, mask):
        ny = obs.shape[2]
        nx = obs.shape[3]
        x0 = torch.cat([obs, coords(obs), mask_map(mask, self.nt, ny, nx, obs.dtype)], dim=1)
        s = self.stem(x0)
        x = F.relu(self.n1(self.c1(s)))
        h1 = F.adaptive_max_pool2d(x, ((ny + 1) // 2, (nx + 1) // 2))
        h1 = F.relu(self.nd1(self.d1(h1)))
        h2 = F.adaptive_max_pool2d(h1, ((ny + 3) // 4, (nx + 3) // 4))
        h2 = F.relu(self.nd2(self.d2(h2)))
        up = F.interpolate(self.u1(h1), size=(ny, nx), mode='nearest') + F.interpolate(self.u2(h2), size=(ny, nx), mode='nearest')
        y = F.relu(x + up)
        f = F.relu(y + self.n2(self.c2(y)))
        p = self.proj(torch.cat([f, self.skip(x0)], dim=1))
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            h = F.relu(mix(fp_pool(p, dy, dx)))
            res.append(out(h))
        place = torch.cat(res, dim=1).flatten(1)
        dm = F.relu(self.det(x0))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), h2.amax((2, 3)), dm.amax((2, 3)), dm.mean((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
