import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
RAW = 16
DET = 16
PROJ = 32
HID = 32
GLOB = 64

RECIPE = {'ppo_epochs': 3, 'n_envs': 256, 'rollout_len': 32, 'n_minibatches': 8, 'lr': 0.0005, 'gamma': 0.98, 'gae_lambda': 0.9, 'ent_coef': 0.01}


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


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.lnp = math.log(float(spec.n_place))
        self.stem = nn.Sequential(
            nn.Conv2d(c + 2, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.GroupNorm(8, COL), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.raw = nn.Conv2d(c, RAW, 1)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.mix1 = nn.ModuleList([nn.Conv2d(3 * (PROJ + RAW), HID, 1) for _ in self.fps])
        self.hn = nn.ModuleList([nn.GroupNorm(1, HID) for _ in self.fps])
        self.mix2 = nn.ModuleList([nn.Conv2d(HID, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.det = nn.Conv2d(c, DET, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + 2 * DET, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB + 2, 1)
        self.value_head = nn.Linear(GLOB, 1)
        init_layers(self)
        for o in self.outs:
            init_out(o, 0.01)
        init_out(self.done_head, 0.01)
        init_out(self.value_head, 1.0)

    def forward(self, obs, mask):
        s = self.stem(torch.cat([obs, coords(obs)], dim=1))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        p = torch.cat([self.proj(f), self.raw(obs)], dim=1)
        res = []
        for (dy, dx), m1, hn, m2, out in zip(self.fps, self.mix1, self.hn, self.mix2, self.outs):
            h = F.relu(hn(m1(fp_pool(p, dy, dx))))
            h = F.relu(m2(h))
            res.append(out(h))
        place = torch.cat(res, dim=1).flatten(1)
        dm = F.relu(self.det(obs))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), dm.amax((2, 3)), dm.mean((2, 3))], dim=1))
        pd = place.detach()
        dctx = torch.cat([g, pd.amax(1, keepdim=True), torch.logsumexp(pd, 1, keepdim=True) - self.lnp], dim=1)
        logits = torch.cat([place, self.done_head(dctx)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
