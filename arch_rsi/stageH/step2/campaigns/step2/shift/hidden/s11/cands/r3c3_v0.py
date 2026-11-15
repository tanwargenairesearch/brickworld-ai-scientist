import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 32, "n_minibatches": 16, "ppo_epochs": 2, "gamma": 0.97}

STEM = 64
COL = 48
WIDTH = 40
LW = 24
HID = 16


def rel_column(v, Z, below):
    p = F.pad(v, (0, 0, 0, 0, Z - 1, 0), value=below)
    p = F.pad(p, (0, 0, 0, 0, 0, Z - 1), value=0.0)
    return torch.stack([p[:, z:z + 2 * Z - 1] for z in range(Z)], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.nlin = 4 * (2 * self.Z - 1) + 4
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c3 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.g1 = nn.Linear(2 * WIDTH, WIDTH)
        self.g2 = nn.Linear(2 * WIDTH, WIDTH)
        self.ctx = nn.Conv2d(WIDTH, LW, 1)
        self.lin = nn.Conv2d(self.nlin, LW, 1)
        self.l2 = nn.Conv2d(LW, LW, 3, padding=1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * LW, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def layer_inputs(self, obs):
        Z = self.Z
        B, C, Y, X = obs.shape
        parts = [rel_column(obs[:, 0:Z], Z, 1.0),
                 rel_column(obs[:, Z:2 * Z], Z, 1.0),
                 rel_column(obs[:, 2 * Z:3 * Z], Z, 0.0),
                 rel_column(obs[:, 3 * Z:4 * Z], Z, 0.0)]
        for i in range(3):
            parts.append(obs[:, 4 * Z + i:4 * Z + i + 1].unsqueeze(1).expand(B, Z, 1, Y, X))
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) * (1.0 / Z)).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        parts.append(zc)
        return torch.cat(parts, dim=2).reshape(B * Z, self.nlin, Y, X)

    def forward(self, obs, mask):
        Z = self.Z
        B, C, Y, X = obs.shape
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        g = self.g1(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        x = F.relu(x + self.c2(x) + g[:, :, None, None])
        g = self.g2(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        f = F.relu(x + self.c3(x) + g[:, :, None, None])
        c = self.ctx(f).unsqueeze(1).expand(B, Z, LW, Y, X).reshape(B * Z, LW, Y, X)
        h = F.relu(self.lin(self.layer_inputs(obs)) + c)
        h = F.relu(h + self.l2(h))
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = h, h, h
            else:
                q = F.pad(h, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            o = out(F.relu(mix(torch.cat([mx, mn, av], dim=1)))).reshape(B, Z, 2, Y, X)
            w = torch.softmax(o[:, :, 1], dim=1)
            res.append((w * o[:, :, 0]).sum(1).flatten(1))
        place = torch.cat(res, dim=1)
        gg = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
