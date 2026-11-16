import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 64, "rollout_len": 64, "n_minibatches": 16,
          "lr": 0.002, "clip_eps": 0.3, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

COLH = 96
COL = 48
CTX = 24
HID = 40
GH = 48


def init_all(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, math.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def init_out(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        z = int(spec.Z)
        c = spec.in_channels
        cells = int(spec.Y) * int(spec.X)
        self.dc = max(2, 512 // cells)
        feat = COL + CTX
        self.col1 = nn.Conv2d(c, COLH, 1)
        self.col2 = nn.Conv2d(COLH, COL, 1)
        self.gn = nn.GroupNorm(8, COL)
        self.ctx = nn.Conv2d(COL, CTX, 3, padding=1)
        self.foot = nn.ModuleList([nn.Conv2d(feat, HID, (dy, dx)) for dy, dx in self.fps])
        self.land = nn.ModuleList([nn.Conv2d(HID, z, 1) for _ in self.fps])
        self.corr = nn.ModuleList([nn.Conv2d(HID, z, 1) for _ in self.fps])
        self.direct = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.anchor = nn.Conv2d(feat, 1, 1)
        self.dproj = nn.Conv2d(feat, self.dc, 1)
        self.dense = nn.Linear(self.dc * cells, spec.n_place)
        self.gproj = nn.Conv2d(feat, GH, 1)
        self.glob = nn.Linear(2 * GH, 64)
        self.type_bias = nn.Linear(64, len(self.fps))
        self.done = nn.Linear(64, 1)
        self.value = nn.Linear(64, 1)
        init_all(self)
        for o in self.land:
            init_out(o, 1.0)
        for o in self.corr:
            init_out(o, 0.01)
        for o in self.direct:
            init_out(o, 0.01)
        init_out(self.anchor, 0.01)
        init_out(self.dense, 0.01)
        init_out(self.type_bias, 0.01)
        init_out(self.done, 0.01)
        init_out(self.value, 1.0)

    def forward(self, obs, mask):
        col = F.relu(self.gn(self.col2(F.relu(self.col1(obs)))))
        f = torch.cat([col, F.relu(self.ctx(col))], dim=1)
        gp = F.relu(self.gproj(f))
        g = F.relu(self.glob(torch.cat([gp.amax((2, 3)), gp.mean((2, 3))], dim=1)))
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            h = F.relu(self.foot[i](q))
            w = torch.softmax(self.land[i](h), dim=1)
            res.append((w * self.corr[i](h)).sum(1, keepdim=True) + self.direct[i](h))
        place = torch.cat(res, dim=1) + self.anchor(f) + self.type_bias(g)[:, :, None, None]
        d = self.dense(F.relu(self.dproj(f)).flatten(1))
        place = place.flatten(1) + d
        logits = torch.cat([place, self.done(g)], dim=1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
