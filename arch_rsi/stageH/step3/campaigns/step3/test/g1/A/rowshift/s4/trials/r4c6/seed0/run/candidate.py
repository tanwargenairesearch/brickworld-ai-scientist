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
BC = 8
HH = 40
LH = 16
LM = 32
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


def d4(r, square):
    copies = [r, r.flip(2), r.flip(3), r.flip(2).flip(3)]
    if square:
        t = r.transpose(2, 3)
        copies = copies + [t, t.flip(2), t.flip(3), t.flip(2).flip(3)]
    return torch.cat(copies, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        nt = len(self.fps)
        self.square = int(spec.Y) == int(spec.X)
        mdy = max(a for a, b in self.fps)
        mdx = max(b for a, b in self.fps)
        ks = (2 * mdy - 1, 2 * mdx - 1)
        pd = (mdy - 1, mdx - 1)
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
        ncopy = 8 if self.square else 4
        self.red = nn.Conv2d(feat, RED, 1)
        self.bc = nn.Linear(GH, BC)
        self.h1 = nn.Conv2d(RED * ncopy + BC, HH, ks, padding=pd)
        self.h2 = nn.Conv2d(HH, HH, 1)
        self.out1 = nn.Conv2d(HH, nt, 1)
        self.loc = nn.Conv2d(feat, LH, ks, padding=pd)
        self.lmix = nn.Conv2d(LH * ncopy, LM, 1)
        self.out2 = nn.Conv2d(LM, nt, 1)
        self.anchor = nn.Conv2d(feat, 1, 1)
        self.act_bias = nn.Parameter(torch.zeros(1, nt, int(spec.Y), int(spec.X)))
        self.glob = nn.Linear(2 * feat, GH)
        self.type_bias = nn.Linear(GH, nt)
        self.done = nn.Linear(GH, 1)
        self.value = nn.Linear(GH, 1)
        init_all(self)
        init_out(self.out1, 0.01)
        init_out(self.out2, 0.01)
        init_out(self.anchor, 0.01)
        init_out(self.type_bias, 0.01)
        init_out(self.done, 0.01)
        init_out(self.value, 1.0)

    def forward(self, obs, mask):
        s = F.relu(self.gn0(self.stem2(F.relu(self.stem1(obs)))))
        x = F.relu(self.gn1(self.c1(s)))
        x = F.relu(x + self.gn2(self.c2(x)))
        f = torch.cat([x, F.relu(self.raw(obs))], dim=1)
        g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)))
        r = self.red(f)
        b = self.bc(g)[:, :, None, None].expand(-1, -1, f.shape[2], f.shape[3])
        h = F.relu(self.h1(torch.cat([d4(r, self.square), b], dim=1)))
        h = F.relu(self.h2(h))
        l = F.relu(self.loc(f))
        m = F.relu(self.lmix(d4(l, self.square)))
        place = self.out1(h) + self.out2(m) + self.anchor(f) + self.type_bias(g)[:, :, None, None] + self.act_bias
        done = DONE_SCALE * self.done(g)
        logits = torch.cat([place.flatten(1), done], dim=1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
