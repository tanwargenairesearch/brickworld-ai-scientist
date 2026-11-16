import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

WIDTH = 32
SKIP = 16
HHID = 32
SCALE = 5.0


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


def frame(x, k):
    if k >= 4:
        x = x.transpose(2, 3)
    r = k % 4
    if r == 1:
        x = x.flip(3)
    elif r == 2:
        x = x.flip(2)
    elif r == 3:
        x = x.flip(2).flip(3)
    return x


class Trunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        g = math.sqrt(2.0)
        self.stem = nn.Sequential(ortho(nn.Conv2d(c, 64, 1), g), nn.ReLU(),
                                  ortho(nn.Conv2d(64, 48, 1), g), nn.ReLU())
        self.c1 = ortho(nn.Conv2d(48, WIDTH, 3, padding=1), g)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = ortho(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), g)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = ortho(nn.Conv2d(c, SKIP, 1), g)

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        h = torch.cat([f, F.relu(self.skip(obs))], 1)
        return f, h


class SquareHead(nn.Module):
    def __init__(self, cin, spec):
        super().__init__()
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(fps)
        self.K = max(max(a, b) for a, b in fps)
        if int(spec.X) == int(spec.Y):
            self.frames = [0, 4]
        else:
            self.frames = [0, 3]
        nf = len(self.frames)
        self.convs = nn.ModuleList([ortho(nn.Conv2d(cin, HHID, self.K), math.sqrt(2.0)) for _ in range(nf)])
        self.outs = nn.ModuleList([ortho(nn.Conv2d(HHID, self.T, 1), 0.01) for _ in range(nf)])
        self.tmix = nn.Parameter(torch.zeros(nf, self.T, self.T))
        self.mixw = nn.Parameter(torch.zeros(nf))

    def forward(self, h):
        K = self.K
        L = []
        for i, k in enumerate(self.frames):
            x = frame(h, k)
            x = F.pad(x, (0, K - 1, 0, K - 1))
            z = F.relu(self.convs[i](x))
            o = self.outs[i](z)
            o = o + torch.einsum('btyx,ts->bsyx', o, self.tmix[i])
            L.append(o.flatten(1))
        L = torch.stack(L, 1)
        a = torch.softmax(SCALE * self.mixw, dim=0)
        return (L * a[None, :, None]).sum(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec)
        self.head = SquareHead(WIDTH + SKIP, spec)
        self.glob = nn.Sequential(ortho(nn.Linear(2 * WIDTH, 64), math.sqrt(2.0)), nn.ReLU())
        self.dense = ortho(nn.Linear(64, spec.n_place), 0.01)
        self.done_head = ortho(nn.Linear(64, 1), 0.01)
        self.value_head = ortho(nn.Linear(64, 1), 1.0)

    def forward(self, obs, mask):
        f, h = self.trunk(obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        place = self.head(h) + self.dense(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
