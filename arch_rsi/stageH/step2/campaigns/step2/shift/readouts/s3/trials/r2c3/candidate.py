import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'n_envs': 128, 'rollout_len': 16, 'n_minibatches': 8, 'ppo_epochs': 3}

TRUNK = 48
WIDTH = 32
LH = 48
LW = 32
PROJ = 16
HID = 24


def align(obs):
    s = obs.shape[3] // 2
    return torch.cat([obs[:, :, :, s:], obs[:, :, :, :s]], dim=3)


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
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = spec.Z
        self.Z = Z
        self.trunk_in = nn.Conv2d(spec.in_channels, TRUNK, 1)
        self.c1 = nn.Conv2d(TRUNK, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.ctx = nn.Conv2d(WIDTH, LH, 1)
        self.l1 = nn.Conv2d(9 + Z, LH, 1)
        self.l2 = nn.Conv2d(LH, LW, 1)
        self.head = PoolHead(LW, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        self.register_buffer('eye', torch.eye(Z))

    def forward(self, obs, mask):
        obs = align(obs)
        B, C, Y, X = obs.shape
        Z = self.Z
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        rest = obs[:, 4 * Z:4 * Z + 3]
        bb = F.pad(b, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        tb = F.pad(t, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        per = torch.stack([b, t, m, e, bb, tb], dim=2)
        rest_e = rest.unsqueeze(1).expand(B, Z, 3, Y, X)
        pos = self.eye.view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        li = torch.cat([per, rest_e, pos], dim=2).reshape(B * Z, 9 + Z, Y, X)

        s = F.relu(self.trunk_in(obs))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))

        ctx = self.ctx(f).unsqueeze(1).expand(B, Z, LH, Y, X).reshape(B * Z, LH, Y, X)
        u = F.relu(self.l1(li) + ctx)
        u = F.relu(self.l2(u))
        place = self.head(u).reshape(B, Z, -1).sum(1)

        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
