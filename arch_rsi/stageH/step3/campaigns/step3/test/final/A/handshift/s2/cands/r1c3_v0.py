import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 40
SKIP = 16
HID = 28
GH = 64
SCALE = 5.0


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = nn.Conv2d(cin, SKIP, 1)

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        f = torch.cat([x, F.relu(self.skip(obs))], dim=1)
        return x, f


class SquareHead(nn.Module):
    def __init__(self, c, spec):
        super().__init__()
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.K = max(max(a, b) for a, b in fps)
        T = len(fps)
        self.convs = nn.ModuleList([nn.Conv2d(c, HID, self.K) for _ in range(2)])
        self.outs = nn.ModuleList([nn.Conv2d(HID, T, 1) for _ in range(2)])
        self.tmix = nn.ModuleList([nn.Conv2d(T, T, 1, bias=False) for _ in range(2)])
        self.alpha = nn.Parameter(torch.zeros(2))

    def forward(self, f):
        K = self.K
        cands = []
        for i in range(2):
            x = f if i == 0 else f.transpose(-1, -2)
            x = F.pad(x, (0, K - 1, 0, K - 1))
            h = F.relu(self.convs[i](x))
            l = self.outs[i](h)
            l = l + self.tmix[i](l)
            cands.append(l.flatten(1))
        L = torch.stack(cands, dim=0)
        w = torch.softmax(SCALE * self.alpha, dim=0)
        return (L * w[:, None, None]).sum(0)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec.in_channels)
        self.head = SquareHead(WIDTH + SKIP, spec)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.dense = nn.Linear(GH, spec.n_place)
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.head.outs:
            nn.init.orthogonal_(m.weight, 0.01)
        for m in self.head.tmix:
            nn.init.zeros_(m.weight)
        nn.init.orthogonal_(self.dense.weight, 0.01)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.orthogonal_(self.value_head.weight, 1.0)

    def forward(self, obs, mask):
        x, f = self.trunk(obs)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        pl = self.head(f) + self.dense(g)
        logits = torch.cat([pl, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
