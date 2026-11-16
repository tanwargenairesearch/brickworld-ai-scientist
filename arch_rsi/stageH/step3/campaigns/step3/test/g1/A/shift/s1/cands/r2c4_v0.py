import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.002, 'n_envs': 256, 'rollout_len': 16, 'n_minibatches': 16, 'ppo_epochs': 2, 'ent_coef': 0.01, 'gamma': 0.95, 'gae_lambda': 0.9, 'clip_eps': 0.3}

STEM = 64
W = 32
SKIP = 32
PA = 32
PC = 32
HID = 32
H2 = 32
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


def fp_pool2(p, dy, dx):
    if dy * dx == 1:
        return p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    return F.max_pool2d(q, (dy, dx), stride=1), -F.max_pool2d(-q, (dy, dx), stride=1)


class GConv(nn.Module):
    def __init__(self, cin, cout, k):
        super().__init__()
        self.c = nn.Conv2d(cin, 2 * cout, k, padding=k // 2)

    def forward(self, x):
        return F.glu(self.c(x), dim=1)


class GBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.GroupNorm(4, c)
        self.g = GConv(c, c, 3)
        self.n2 = nn.GroupNorm(4, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        h = self.g(F.relu(self.n1(x)))
        h = self.c2(F.relu(self.n2(h)))
        return x + h


class RelHead(nn.Module):
    def __init__(self, cin, fps):
        super().__init__()
        self.fps = fps
        self.pa = nn.Conv2d(cin, PA, 1)
        self.pc = nn.Conv2d(cin, PC, 1)
        self.la = nn.ModuleList([nn.Conv2d(2 * PA, HID, 1) for _ in fps])
        self.lc = nn.ModuleList([nn.Conv2d(PC, HID, 1, bias=False) for _ in fps])
        self.l2 = nn.ModuleList([nn.Conv2d(HID, 2 * H2, 1) for _ in fps])
        self.outs = nn.ModuleList([nn.Conv2d(2 * H2, 1, 1) for _ in fps])

    def forward(self, f):
        ny, nx = f.shape[2], f.shape[3]
        a = self.pa(f)
        c = self.pc(f)
        res = []
        for k, (dy, dx) in enumerate(self.fps):
            amx, amn = fp_pool2(a, dy, dx)
            ha = self.la[k](torch.cat([amx, amn], dim=1))
            hc = F.pad(self.lc[k](c), (0, dx - 1, 0, dy - 1))
            zs = []
            for i in range(dy):
                for j in range(dx):
                    h = F.relu(ha + hc[:, :, i:i + ny, j:j + nx])
                    zs.append(F.glu(self.l2[k](h), dim=1))
            z = torch.stack(zs, dim=0)
            res.append(self.outs[k](torch.cat([z.amin(0), z.mean(0)], dim=1)))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem1 = GConv(cin + 2, STEM, 1)
        self.stem2 = GConv(STEM, W, 1)
        self.block = GBlock(W)
        self.nf = nn.GroupNorm(4, W)
        self.skip = nn.Conv2d(cin, SKIP, 1)
        self.head = RelHead(W + SKIP, fps)
        self.glob = nn.Sequential(nn.Linear(2 * (W + SKIP), G), nn.ReLU(), nn.Linear(G, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Sequential(nn.Linear(G, G), nn.ReLU(), nn.Linear(G, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                orth(m, math.sqrt(2.0))
        for m in self.head.outs:
            orth(m, 0.01)
        orth(self.done_head, 0.01)
        orth(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        x = self.stem2(self.stem1(torch.cat([obs, coord_channels(obs)], dim=1)))
        x = self.block(x)
        f = torch.cat([F.relu(self.nf(x)), F.relu(self.skip(obs))], dim=1)
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
