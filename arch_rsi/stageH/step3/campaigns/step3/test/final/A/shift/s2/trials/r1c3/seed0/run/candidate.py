import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 32
SKIP = 16
GH = 64
HID = 48
SCALE = 5.0


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(self.fps)
        self.nf = 2 if spec.Y == spec.X else 1
        self.S = max(max(a, b) for a, b in self.fps)
        self.stem1 = nn.Conv2d(c, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, COL, 1)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.sq = nn.ModuleList([nn.Conv2d(WIDTH + SKIP, HID, self.S) for _ in range(self.nf)])
        self.outs = nn.ModuleList([nn.Conv2d(HID, self.T, 1) for _ in range(self.nf)])
        self.glob = nn.Linear(2 * WIDTH, GH)
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)
        self.type_bias = nn.Linear(GH, self.T)
        self.dense = nn.Linear(GH, spec.n_actions)
        for mod in self.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                init_layer(mod, math.sqrt(2))
        for o in self.outs:
            init_layer(o, 0.01)
        init_layer(self.done_head, 0.01)
        init_layer(self.type_bias, 0.01)
        init_layer(self.dense, 0.01)
        init_layer(self.value_head, 1.0)
        self.tmix = nn.Parameter(torch.zeros(self.nf, self.T, self.T))
        self.alpha = nn.Parameter(torch.zeros(self.nf))

    def forward(self, obs, mask):
        s = F.relu(self.stem2(F.relu(self.stem1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        feat = torch.cat([f, F.relu(self.skip(obs))], 1)
        g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)))
        tb = self.type_bias(g)[:, :, None, None]
        S = self.S
        cands = []
        for k in range(self.nf):
            fk = feat if k == 0 else feat.transpose(2, 3)
            h = F.relu(self.sq[k](F.pad(fk, (0, S - 1, 0, S - 1))))
            o = self.outs[k](h) + tb
            o = o + torch.einsum('btyx,ts->bsyx', o, self.tmix[k])
            cands.append(o.flatten(1))
        L = torch.stack(cands, 1)
        w = torch.softmax(SCALE * self.alpha, 0)
        place = torch.einsum('bkn,k->bn', L, w)
        logits = torch.cat([place, self.done_head(g)], 1) + self.dense(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
