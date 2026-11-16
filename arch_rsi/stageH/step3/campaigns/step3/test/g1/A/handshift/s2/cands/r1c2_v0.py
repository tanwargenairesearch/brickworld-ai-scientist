import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.003, 'n_envs': 64, 'rollout_len': 64, 'n_minibatches': 32, 'ppo_epochs': 1, 'gamma': 0.995, 'gae_lambda': 0.99, 'clip_eps': 0.3, 'ent_coef': 0.01}

STEM = 64
COL = 48
SK = 32
PROJ = 32
HID = 48
G = 64


def ortho(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = spec.in_channels
        T = len(self.fps)
        self.stem1 = ortho(nn.Conv2d(C, STEM, 1))
        self.stem2 = ortho(nn.Conv2d(STEM, COL, 1))
        self.conv = ortho(nn.Conv2d(COL, COL, 3, padding=1), 1.0)
        self.norm = nn.GroupNorm(4, COL)
        self.skip = ortho(nn.Conv2d(C, SK, 1))
        self.proj = ortho(nn.Conv2d(COL + SK, PROJ, 1), 1.0)
        self.fpc = nn.ModuleList([ortho(nn.Conv2d(PROJ, HID, (dy, dx))) for dy, dx in self.fps])
        self.pmix = nn.ModuleList([ortho(nn.Conv2d(2 * PROJ, HID, 1)) for t in range(T)])
        self.outs = nn.ModuleList([ortho(nn.Conv2d(HID, 1, 1), 0.01) for t in range(T)])
        self.anchor = ortho(nn.Conv2d(COL + SK, 1, 1), 0.01)
        self.type_bias = nn.Parameter(torch.zeros(T))
        self.glob = nn.Sequential(ortho(nn.Linear(2 * (COL + SK), G)), nn.ReLU(), ortho(nn.Linear(G, G)), nn.ReLU())
        self.done = ortho(nn.Linear(G, 1), 0.01)
        self.value = ortho(nn.Linear(G, 1), 1.0)

    def forward(self, obs, mask):
        s = F.relu(self.stem2(F.relu(self.stem1(obs))))
        f = F.relu(self.norm(s + self.conv(s)))
        k = F.relu(self.skip(obs))
        h = torch.cat([f, k], dim=1)
        p = self.proj(h)
        a = self.anchor(h)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            c = self.fpc[i](q)
            if dy * dx == 1:
                mx, mn = p, p
            else:
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            z = F.relu(c + self.pmix[i](torch.cat([mx, mn], dim=1)))
            res.append(self.outs[i](z) + a + self.type_bias[i])
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done(g)], dim=1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
