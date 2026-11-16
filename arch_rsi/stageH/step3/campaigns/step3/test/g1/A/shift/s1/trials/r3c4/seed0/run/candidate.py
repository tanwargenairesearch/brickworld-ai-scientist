import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.0015, 'n_envs': 128, 'rollout_len': 32, 'n_minibatches': 16, 'ppo_epochs': 2, 'ent_coef': 0.01, 'gamma': 0.98, 'gae_lambda': 0.95, 'clip_eps': 0.2}

E1 = 64
E = 32
SKIP = 24
FU = 40
PROJ = 32
HID = 32
G = 64


def orth(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def coord_channels(obs):
    b, c, y, x = obs.shape
    ys = torch.linspace(-1.0, 1.0, y, device=obs.device, dtype=obs.dtype).view(1, 1, y, 1).expand(b, 1, y, x)
    xs = torch.linspace(-1.0, 1.0, x, device=obs.device, dtype=obs.dtype).view(1, 1, 1, x).expand(b, 1, y, x)
    return torch.cat([ys, xs], dim=1)


def fp_pool(p, dy, dx):
    if dy * dx == 1:
        return p, p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


class Enc(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.l1 = nn.Conv2d(cin, E1, 1)
        self.l2 = nn.Conv2d(E1, E, 1)
        self.c3 = nn.Conv2d(E, E, 3, padding=1)
        self.n = nn.GroupNorm(4, E)

    def forward(self, x):
        h = F.relu(self.l2(F.relu(self.l1(x))))
        return F.relu(self.n(h + self.c3(h)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.enc = Enc(self.Z + 3)
        self.skip = nn.Conv2d(cin + 2, SKIP, 1)
        self.fuse = nn.Conv2d(4 * E + SKIP, FU, 1)
        self.cc = nn.Conv2d(FU, FU, 3, padding=1)
        self.nc = nn.GroupNorm(4, FU)
        self.proj = nn.Conv2d(FU + SKIP, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * (FU + SKIP), G), nn.ReLU(), nn.Linear(G, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Sequential(nn.Linear(G, G), nn.ReLU(), nn.Linear(G, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                orth(m, math.sqrt(2.0))
        for m in self.outs:
            orth(m, 0.01)
        orth(self.done_head, 0.01)
        orth(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Z = self.Z
        xy = coord_channels(obs)
        bi = torch.cat([obs[:, 0:Z], obs[:, 4 * Z:4 * Z + 1], xy], dim=1)
        ti = torch.cat([obs[:, Z:2 * Z], obs[:, 4 * Z + 1:4 * Z + 2], xy], dim=1)
        e = self.enc(torch.cat([bi, ti], dim=0))
        eb = e[:B]
        et = e[B:]
        sk = F.relu(self.skip(torch.cat([obs, xy], dim=1)))
        h = F.relu(self.fuse(torch.cat([eb, et, et - eb, eb * et, sk], dim=1)))
        h = F.relu(self.nc(h + self.cc(h)))
        f = torch.cat([h, sk], dim=1)
        p = self.proj(f)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            mx, mn, av = fp_pool(p, dy, dx)
            z = F.relu(self.mix[i](torch.cat([mx, mn, av], dim=1)))
            res.append(self.outs[i](z))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
