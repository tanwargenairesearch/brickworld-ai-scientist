import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 36
SKIP = 16
HID = 32
GLOB = 64
SCALE = 5.0


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class Trunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.stem = nn.Sequential(nn.Conv2d(c, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        h = torch.cat([f, self.skip(obs)], dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        return h, g


class SquareHead(nn.Module):
    def __init__(self, c, K, T):
        super().__init__()
        self.K = K
        self.conv = nn.Conv2d(c, HID, K)
        self.out = nn.Conv2d(HID, T, 1)

    def forward(self, h):
        q = F.pad(h, (0, self.K - 1, 0, self.K - 1))
        return self.out(F.relu(self.conv(q)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(fps)
        K = max(max(a, b) for a, b in fps)
        self.square = (Y == X)
        self.nf = 2
        self.trunk = Trunk(spec)
        self.heads = nn.ModuleList([SquareHead(WIDTH + SKIP, K, T) for _ in range(self.nf)])
        self.tmix = nn.Parameter(torch.zeros(self.nf, T, T))
        self.dense = nn.Linear(GLOB, int(spec.n_actions))
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        self.frame_logit = nn.Parameter(torch.zeros(self.nf))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                ortho(m, math.sqrt(2))
        for hd in self.heads:
            ortho(hd.out, 0.01)
        ortho(self.dense, 0.01)
        ortho(self.done_head, 0.01)
        ortho(self.value_head, 1.0)

    def forward(self, obs, mask):
        h, g = self.trunk(obs)
        outs = []
        for i, head in enumerate(self.heads):
            if i == 0:
                hi = h
            elif self.square:
                hi = h.transpose(2, 3)
            else:
                hi = torch.flip(h, (2, 3))
            L = head(hi)
            L = L + torch.einsum('btyx,st->bsyx', L, self.tmix[i])
            outs.append(L.flatten(1))
        per = torch.stack(outs, dim=1)
        w = torch.softmax(SCALE * self.frame_logit, 0)
        place = torch.einsum('bfp,f->bp', per, w)
        logits = torch.cat([place, self.done_head(g)], dim=1) + self.dense(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
