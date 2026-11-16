import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 3, "n_envs": 64, "rollout_len": 32, "n_minibatches": 8,
          "gamma": 0.99, "gae_lambda": 0.95, "lr": 0.0007, "ent_coef": 0.003}

C1 = 128
C2 = 64
COL = 48
CTX = 24
HID = 40
MID = 32
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
        c = spec.in_channels
        cells = int(spec.Y) * int(spec.X)
        self.dc = max(2, 512 // cells)
        feat = COL + CTX
        self.col1 = nn.Conv2d(c, C1, 1)
        self.col2 = nn.Conv2d(C1, C2, 1)
        self.col3 = nn.Conv2d(C2, COL, 1)
        self.ctx = nn.Conv2d(COL, CTX, 3, padding=1)
        self.foot = nn.ModuleList([nn.Conv2d(feat, HID, (dy, dx)) for dy, dx in self.fps])
        self.mid = nn.ModuleList([nn.Conv2d(HID, MID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(MID, 1, 1) for _ in self.fps])
        self.anchor = nn.Conv2d(feat, 1, 1)
        self.dproj = nn.Conv2d(feat, self.dc, 1)
        self.dense = nn.Linear(self.dc * cells, spec.n_place)
        self.gproj = nn.Conv2d(feat, GH, 1)
        self.glob = nn.Linear(2 * GH, 64)
        self.type_bias = nn.Linear(64, len(self.fps))
        self.done = nn.Linear(64, 1)
        self.value = nn.Linear(64, 1)
        init_all(self)
        for o in self.outs:
            init_out(o, 0.01)
        init_out(self.anchor, 0.01)
        init_out(self.dense, 0.01)
        init_out(self.type_bias, 0.01)
        init_out(self.done, 0.01)
        init_out(self.value, 1.0)

    def forward(self, obs, mask):
        h = F.relu(self.col1(obs))
        h = F.relu(self.col2(h))
        col = F.relu(self.col3(h))
        f = torch.cat([col, F.relu(self.ctx(col))], dim=1)
        gp = F.relu(self.gproj(f))
        g = F.relu(self.glob(torch.cat([gp.amax((2, 3)), gp.mean((2, 3))], dim=1)))
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            z = F.relu(self.foot[i](q))
            z = F.relu(self.mid[i](z))
            res.append(self.outs[i](z))
        place = torch.cat(res, dim=1) + self.anchor(f) + self.type_bias(g)[:, :, None, None]
        d = self.dense(F.relu(self.dproj(f)).flatten(1))
        place = place.flatten(1) + d
        logits = torch.cat([place, self.done(g)], dim=1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
