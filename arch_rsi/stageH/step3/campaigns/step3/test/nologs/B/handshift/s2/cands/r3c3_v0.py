import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.98, "gae_lambda": 0.95, "ent_coef": 0.005, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16}

W = 16
PROJ = 12
HID = 16
NBLOCKS = 2


class ChanNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = ChanNorm(c)
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = ChanNorm(c)
        self.zmix = nn.Conv2d(3 * c, c, 1)

    def forward(self, x, B, Z):
        x = x + self.conv(F.relu(self.n1(x)))
        h = F.relu(self.n2(x))
        n, c, Y, X = h.shape
        h5 = h.reshape(B, Z, c, Y, X)
        below = F.pad(h5, (0, 0, 0, 0, 0, 0, 1, 0))[:, :Z]
        above = F.pad(h5, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:]
        hz = torch.cat([below, h5, above], dim=2).reshape(B * Z, 3 * c, Y, X)
        return x + self.zmix(hz)


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.T = int(spec.n_types)
        cin = 9 + self.Z
        self.stem = nn.Sequential(nn.Conv2d(cin, W, 1), nn.ReLU(), nn.Conv2d(W, W, 1))
        self.blocks = nn.ModuleList([Block(W) for _ in range(NBLOCKS)])
        self.norm = ChanNorm(W)
        self.head = PoolHead(W, spec.footprints, PROJ, HID)
        gin = 2 * W + self.Z * W
        self.done_head = nn.Sequential(nn.Linear(gin, 64), nn.ReLU(), nn.Linear(64, 1))
        self.value_head = nn.Sequential(nn.Linear(gin, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.normal_(self.done_head[2].weight, std=0.01)
        nn.init.zeros_(self.done_head[2].bias)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        cur = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        ext = obs[:, 3 * Z:4 * Z]
        curb = F.pad(cur, (0, 0, 0, 0, 1, 0))[:, :Z]
        tgtb = F.pad(tgt, (0, 0, 0, 0, 1, 0))[:, :Z]
        per = torch.stack([cur, tgt, miss, ext, curb, tgtb], dim=2)
        glob = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        pos = torch.eye(Z, dtype=obs.dtype, device=obs.device).reshape(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        inp = torch.cat([per, glob, pos], dim=2).reshape(B * Z, 9 + Z, Y, X)
        x = self.stem(inp)
        for blk in self.blocks:
            x = blk(x, B, Z)
        f = F.relu(self.norm(x))
        s = self.head(f).reshape(B, Z, self.T, Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        f5 = f.reshape(B, Z, W, Y, X)
        g = torch.cat([f5.mean((1, 3, 4)), f5.amax((1, 3, 4)), f5.mean((3, 4)).flatten(1)], dim=1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
