import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "ppo_epochs": 3, "rollout_len": 32, "n_minibatches": 32,
          "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01}

COL = 48
W = 40
SK = 32
HID = 32


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.s1 = nn.Conv2d(C, 64, 1)
        self.s2 = nn.Conv2d(64, COL, 1)
        self.n0 = nn.GroupNorm(1, COL)
        self.c1 = nn.Conv2d(COL, W, 3, padding=1)
        self.n1 = nn.GroupNorm(1, W)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.n2 = nn.GroupNorm(1, W)
        self.skip = nn.Conv2d(C, SK, 1)
        U = W + SK
        self.fconv = nn.ModuleList([nn.Conv2d(U, HID, (dy, dx)) for dy, dx in self.fps])
        self.hid = nn.ModuleList([nn.Conv2d(HID, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.anchor = nn.Conv2d(U, 1, 1)
        self.glob = nn.Linear(2 * W + 2 * SK, 64)
        self.type_bias = nn.Linear(64, len(self.fps))
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                ortho(m, math.sqrt(2))
        for m in self.outs:
            ortho(m, 0.01)
        ortho(self.anchor, 0.01)
        ortho(self.type_bias, 0.01)
        ortho(self.done_head, 0.01)
        ortho(self.value_head, 1.0)

    def forward(self, obs, mask):
        s = F.relu(self.skip(obs))
        h = F.relu(self.n0(self.s2(F.relu(self.s1(obs)))))
        x = F.relu(self.n1(self.c1(h)))
        f = F.relu(x + self.n2(self.c2(x)))
        u = torch.cat([f, s], dim=1)
        a = self.anchor(u)
        g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)),
                                         s.mean((2, 3)), s.amax((2, 3))], dim=1)))
        tb = self.type_bias(g)
        res = []
        for i, ((dy, dx), fc, hd, out) in enumerate(zip(self.fps, self.fconv, self.hid, self.outs)):
            q = F.pad(u, (0, dx - 1, 0, dy - 1))
            hh = F.relu(fc(q))
            hh = F.relu(hd(hh))
            res.append(out(hh) + a + tb[:, i].view(-1, 1, 1, 1))
        place = torch.cat(res, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
