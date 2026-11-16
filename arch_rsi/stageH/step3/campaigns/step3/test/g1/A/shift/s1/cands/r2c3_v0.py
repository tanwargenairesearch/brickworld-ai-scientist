import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.001, 'n_envs': 128, 'rollout_len': 32, 'n_minibatches': 16, 'ppo_epochs': 3, 'ent_coef': 0.01, 'gamma': 0.97, 'gae_lambda': 0.95, 'clip_eps': 0.2}

VC = 24
UC = 16
W = 32
PROJ = 24
HID = 24
LH = 16
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


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.GroupNorm(4, c)
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(4, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.relu(self.n1(x)))
        h = self.c2(F.relu(self.n2(h)))
        return x + h


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.v1 = nn.Conv2d(8, VC, 1)
        self.v2 = nn.Conv2d(VC, VC, 1)
        self.vz = nn.Conv2d(3 * VC, UC, 1)
        self.flat = nn.Conv2d(UC * self.Z + cin + 2, W, 1)
        self.block = Block(W)
        self.nf = nn.GroupNorm(4, W)
        self.proj = nn.Conv2d(W, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.lctx = nn.ModuleList([nn.Conv2d(2 * PROJ, LH, 1) for _ in self.fps])
        self.lmix = nn.ModuleList([nn.Conv2d(2 * UC, LH, 1) for _ in self.fps])
        self.lout = nn.ModuleList([nn.Conv2d(LH, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W + UC, G), nn.ReLU(), nn.Linear(G, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Sequential(nn.Linear(G, G), nn.ReLU(), nn.Linear(G, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                orth(m, math.sqrt(2.0))
        for m in self.outs:
            orth(m, 0.01)
        for m in self.lout:
            orth(m, 0.01)
        orth(self.done_head, 0.01)
        orth(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        B, _, ny, nx = obs.shape
        Z = self.Z
        vox = obs[:, :4 * Z].reshape(B, 4, Z, ny, nx)
        hz = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, ny, nx)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, 1, Z, 1, 1).expand(B, 1, Z, ny, nx)
        inp = torch.cat([vox, hz, zc], dim=1).reshape(B, 8, Z * ny, nx)
        v = F.relu(self.v2(F.relu(self.v1(inp))))
        v5 = v.reshape(B, VC, Z, ny, nx)
        below = F.pad(v5, (0, 0, 0, 0, 1, 0))[:, :, :Z]
        above = F.pad(v5, (0, 0, 0, 0, 0, 1))[:, :, 1:]
        u = F.relu(self.vz(torch.cat([v5, below, above], dim=1).reshape(B, 3 * VC, Z * ny, nx)))
        uf = u.reshape(B, UC * Z, ny, nx)
        x = self.flat(torch.cat([uf, obs, coord_channels(obs)], dim=1))
        x = self.block(x)
        f = F.relu(self.nf(x))
        p = self.proj(f)
        res = []
        for k, (dy, dx) in enumerate(self.fps):
            if dy * dx == 1:
                mx, mn, av = p, p, p
                umx, umn = uf, uf
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
                uq = F.pad(uf, (0, dx - 1, 0, dy - 1))
                umx = F.max_pool2d(uq, (dy, dx), stride=1)
                umn = -F.max_pool2d(-uq, (dy, dx), stride=1)
            h = F.relu(self.mix[k](torch.cat([mx, mn, av], dim=1)))
            s2d = self.outs[k](h)
            ctx = self.lctx[k](torch.cat([mx, mn], dim=1))
            lu = torch.cat([umx.reshape(B, UC, Z * ny, nx), umn.reshape(B, UC, Z * ny, nx)], dim=1)
            lh = self.lmix[k](lu).reshape(B, LH, Z, ny, nx) + ctx.unsqueeze(2)
            lo = self.lout[k](F.relu(lh).reshape(B, LH, Z * ny, nx)).reshape(B, 2, Z, ny, nx)
            w = torch.softmax(lo[:, 0], dim=1)
            slay = (w * lo[:, 1]).sum(1, keepdim=True)
            res.append(s2d + slay)
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), u.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
