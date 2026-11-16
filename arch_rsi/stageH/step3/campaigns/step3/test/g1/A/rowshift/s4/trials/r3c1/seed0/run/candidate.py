import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 32, "n_minibatches": 32,
          "gamma": 0.97, "gae_lambda": 0.95, "lr": 0.001, "ent_coef": 0.01}

STEM = 64
COL = 48
W = 40
RAW = 16
RED = 8
HH = 56
GH = 64
DONE_SCALE = 0.1


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
        nt = len(self.fps)
        self.square = int(spec.Y) == int(spec.X)
        mdy = max(a for a, b in self.fps)
        mdx = max(b for a, b in self.fps)
        c = spec.in_channels
        self.stem1 = nn.Conv2d(c, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, COL, 1)
        self.gn0 = nn.GroupNorm(8, COL)
        self.c1 = nn.Conv2d(COL, W, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, W)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, W)
        self.raw = nn.Conv2d(c, RAW, 1)
        feat = W + RAW
        self.red = nn.Conv2d(feat, RED, 1)
        ncopy = 8 if self.square else 4
        self.h1 = nn.Conv2d(RED * ncopy, HH, (2 * mdy - 1, 2 * mdx - 1), padding=(mdy - 1, mdx - 1))
        self.h2 = nn.Conv2d(HH, HH, 1)
        self.out = nn.Conv2d(HH, nt, 1)
        self.anchor = nn.Conv2d(feat, 1, 1)
        self.act_bias = nn.Parameter(torch.zeros(1, nt, int(spec.Y), int(spec.X)))
        self.glob = nn.Linear(2 * feat, GH)
        self.type_bias = nn.Linear(GH, nt)
        self.done = nn.Linear(GH, 1)
        self.value = nn.Linear(GH, 1)
        init_all(self)
        init_out(self.out, 0.01)
        init_out(self.anchor, 0.01)
        init_out(self.type_bias, 0.01)
        init_out(self.done, 0.01)
        init_out(self.value, 1.0)

    def forward(self, obs, mask):
        s = F.relu(self.gn0(self.stem2(F.relu(self.stem1(obs)))))
        x = F.relu(self.gn1(self.c1(s)))
        x = F.relu(x + self.gn2(self.c2(x)))
        f = torch.cat([x, F.relu(self.raw(obs))], dim=1)
        r = self.red(f)
        copies = [r, r.flip(2), r.flip(3), r.flip(2).flip(3)]
        if self.square:
            t = r.transpose(2, 3)
            copies = copies + [t, t.flip(2), t.flip(3), t.flip(2).flip(3)]
        h = F.relu(self.h1(torch.cat(copies, dim=1)))
        h = F.relu(self.h2(h))
        g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)))
        place = self.out(h) + self.anchor(f) + self.type_bias(g)[:, :, None, None] + self.act_bias
        done = DONE_SCALE * self.done(g)
        logits = torch.cat([place.flatten(1), done], dim=1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
