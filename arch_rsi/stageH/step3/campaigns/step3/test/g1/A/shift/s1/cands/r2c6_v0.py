import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.0015, 'n_envs': 256, 'rollout_len': 32, 'n_minibatches': 16, 'ppo_epochs': 1, 'ent_coef': 0.01, 'gamma': 0.98, 'gae_lambda': 0.95, 'clip_eps': 0.2}

VC = 24
UC = 16
W = 32
SKIP = 32
PA = 24
PC = 24
HID = 24
H2 = 24
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


def fp_pool2(p, dy, dx):
    if dy * dx == 1:
        return p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    return F.max_pool2d(q, (dy, dx), stride=1), -F.max_pool2d(-q, (dy, dx), stride=1)


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
        t = len(self.fps)
        self.v1 = nn.Conv2d(8, VC, 1)
        self.v2 = nn.Conv2d(VC, VC, 1)
        self.vz = nn.Conv2d(3 * VC, UC, 1)
        self.flat = nn.Conv2d(UC * self.Z + cin + 2, 2 * W, 1)
        self.block = Block(W)
        self.nf = nn.GroupNorm(4, W)
        self.skip = nn.Conv2d(cin, SKIP, 1)
        fc = W + SKIP
        self.pa = nn.Conv2d(fc, PA, 1)
        self.pc = nn.Conv2d(fc, PC, 1)
        self.la = nn.ModuleList([nn.Conv2d(2 * PA, HID, 1) for _ in self.fps])
        self.lc = nn.ModuleList([nn.Conv2d(PC, HID, 1, bias=False) for _ in self.fps])
        self.l2 = nn.ModuleList([nn.Conv2d(HID, 2 * H2, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(2 * H2, 1, 1) for _ in self.fps])
        self.lctx = nn.ModuleList([nn.Conv2d(2 * PA, LH, 1) for _ in self.fps])
        self.lmix = nn.ModuleList([nn.Conv2d(2 * UC, LH, 1) for _ in self.fps])
        self.lout = nn.ModuleList([nn.Conv2d(LH, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * fc + UC, G), nn.ReLU(), nn.Linear(G, G), nn.ReLU())
        self.type_bias = nn.Linear(G, t)
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Sequential(nn.Linear(G, G), nn.ReLU(), nn.Linear(G, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                orth(m, math.sqrt(2.0))
        for m in self.outs:
            orth(m, 0.01)
        for m in self.lout:
            orth(m, 0.01)
        orth(self.type_bias, 0.01)
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
        x = F.glu(self.flat(torch.cat([uf, obs, coord_channels(obs)], dim=1)), dim=1)
        x = self.block(x)
        f = torch.cat([F.relu(self.nf(x)), F.relu(self.skip(obs))], dim=1)
        a = self.pa(f)
        c = self.pc(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), u.amax((2, 3))], dim=1))
        tb = self.type_bias(g)
        res = []
        for k, (dy, dx) in enumerate(self.fps):
            amx, amn = fp_pool2(a, dy, dx)
            A = torch.cat([amx, amn], dim=1)
            ha = self.la[k](A)
            hc = F.pad(self.lc[k](c), (0, dx - 1, 0, dy - 1))
            zs = []
            for i in range(dy):
                for j in range(dx):
                    h = F.relu(ha + hc[:, :, i:i + ny, j:j + nx])
                    zs.append(F.glu(self.l2[k](h), dim=1))
            z = torch.stack(zs, dim=0)
            rel = self.outs[k](torch.cat([z.amin(0), z.mean(0)], dim=1))
            umx, umn = fp_pool2(uf, dy, dx)
            lu = torch.cat([umx.reshape(B, UC, Z * ny, nx), umn.reshape(B, UC, Z * ny, nx)], dim=1)
            lh = self.lmix[k](lu).reshape(B, LH, Z, ny, nx) + self.lctx[k](A).unsqueeze(2)
            lo = self.lout[k](F.relu(lh).reshape(B, LH, Z * ny, nx)).reshape(B, 2, Z, ny, nx)
            slay = (torch.softmax(lo[:, 0], dim=1) * lo[:, 1]).sum(1, keepdim=True)
            res.append(rel + slay + tb[:, k].view(-1, 1, 1, 1))
        place = torch.cat(res, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
