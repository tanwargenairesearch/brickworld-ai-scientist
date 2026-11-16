import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 4, "n_envs": 64, "rollout_len": 16, "n_minibatches": 4, "ent_coef": 0.005}

COL = 32
FEAT = 32
PROJ = 16
HID = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = int(spec.Z)
        self.Z = Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = spec.in_channels
        self.col = nn.Sequential(nn.Conv2d(C, COL, 1), nn.ReLU(),
                                 nn.Conv2d(COL, COL, 1), nn.ReLU())
        self.sp = nn.Conv2d(COL, COL, 3, padding=1)
        self.z1 = nn.Conv2d(COL + 6 + Z, FEAT, 1)
        self.z2 = nn.Conv2d(FEAT, FEAT, 1)
        self.proj = nn.Conv2d(FEAT, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * COL + 2 * FEAT, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        self.register_buffer("zoh", torch.eye(Z).view(1, Z, Z, 1, 1))

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        Z = self.Z
        c = self.col(obs)
        c = F.relu(c + self.sp(c))
        build = obs[:, 0:Z]
        targ = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        bb = F.pad(build[:, :Z - 1], (0, 0, 0, 0, 1, 0), value=1.0)
        tb = F.pad(targ[:, :Z - 1], (0, 0, 0, 0, 1, 0), value=1.0)
        local = torch.stack([build, targ, miss, extra, bb, tb], dim=1)
        cz = c.unsqueeze(2).expand(B, COL, Z, Y, X)
        zo = self.zoh.expand(B, Z, Z, Y, X)
        h = torch.cat([cz, local, zo], dim=1)
        h = h.reshape(B, COL + 6 + Z, Z * Y, X)
        h = F.relu(self.z1(h))
        h = F.relu(h + self.z2(h))
        hz = h.reshape(B, FEAT, Z, Y, X).permute(0, 2, 1, 3, 4).reshape(B * Z, FEAT, Y, X)
        p = self.proj(hz)
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
        s = torch.cat(res, dim=1).reshape(B, Z, len(self.fps), Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        g = torch.cat([c.mean((2, 3)), c.amax((2, 3)), h.mean((2, 3)), h.amax((2, 3))], dim=1)
        g = self.glob(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
