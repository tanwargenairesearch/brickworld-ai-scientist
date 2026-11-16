import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.98, "gae_lambda": 0.95, "ent_coef": 0.005}

TW = 48
LH = 64
LP = 16
HID = 24


def small_init(m):
    nn.init.normal_(m.weight, std=0.01)
    nn.init.zeros_(m.bias)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(1, c)
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.mix = nn.Conv2d(c, c, 1)

    def forward(self, x):
        h = self.conv(F.relu(self.norm(x)))
        return x + self.mix(F.relu(h))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, 64, 1), nn.ReLU(),
            nn.Conv2d(64, TW, 1))
        self.block = Block(TW)
        self.tnorm = nn.GroupNorm(1, TW)
        self.lay1 = nn.Conv2d(11 + Z, LH, 1)
        self.ctx = nn.Conv2d(TW, LH, 1)
        self.lay2 = nn.Conv2d(LH, LP, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * LP, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        for o in self.outs:
            small_init(o)
        self.glob = nn.Sequential(nn.Linear(2 * TW, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        small_init(self.done_head)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        cur = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        mis = obs[:, 2 * Z:3 * Z]
        ext = obs[:, 3 * Z:4 * Z]
        cur_b = F.pad(cur, (0, 0, 0, 0, 1, 0))[:, :Z]
        mis_b = F.pad(mis, (0, 0, 0, 0, 1, 0))[:, :Z]
        tgt_a = F.pad(tgt, (0, 0, 0, 0, 0, 1))[:, 1:]
        mis_a = F.pad(mis, (0, 0, 0, 0, 0, 1))[:, 1:]
        per = torch.stack([cur, tgt, mis, ext, cur_b, mis_b, tgt_a, mis_a], dim=2)
        gin = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zp = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        sl = torch.cat([per, gin, zp], dim=2).reshape(B * Z, 11 + Z, Y, X)

        t = self.block(self.stem(obs))
        f = F.relu(self.tnorm(t))
        c = self.ctx(f).unsqueeze(1).expand(B, Z, LH, Y, X).reshape(B * Z, LH, Y, X)
        h = F.relu(self.lay1(sl) + c)
        p = self.lay2(h)

        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            hh = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(hh))
        s = torch.cat(res, dim=1)
        T = s.shape[1]
        s = s.view(B, Z, T, Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)

        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
