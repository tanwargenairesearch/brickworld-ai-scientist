import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32,
          "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

WIDTH = 32
SKIP = 16
PROJ = 16
HID = 12
GHID = 64
SCALE = 2.0


def frame(f, k):
    if k >= 4:
        f = f.transpose(2, 3)
    if k % 2 == 1:
        f = torch.flip(f, dims=[2])
    if (k // 2) % 2 == 1:
        f = torch.flip(f, dims=[3])
    return f


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain=gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class Trunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.stem1 = nn.Conv2d(c, 64, 1)
        self.stem2 = nn.Conv2d(64, 48, 1)
        self.c1 = nn.Conv2d(48, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.out_ch = WIDTH + SKIP
        for m in [self.stem1, self.stem2, self.c1, self.c2, self.skip]:
            init_layer(m, math.sqrt(2))

    def forward(self, obs):
        s = F.relu(self.stem2(F.relu(self.stem1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        sk = F.relu(self.skip(obs))
        return torch.cat([x, sk], dim=1), x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = spec.Y
        self.X = spec.X
        self.T = spec.n_types
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.nf = 8 if spec.Y == spec.X else 4
        self.trunk = Trunk(spec)
        self.proj = nn.Conv2d(self.trunk.out_ch, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for t in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for t in self.fps])
        self.glob = nn.Linear(2 * WIDTH, GHID)
        self.done_head = nn.Linear(GHID, 1)
        self.value_head = nn.Linear(GHID, 1)
        self.type_bias = nn.Linear(GHID, self.T)
        self.frame_logit = nn.Parameter(torch.zeros(self.T, self.nf))
        init_layer(self.proj, 1.0)
        for m in self.mix:
            init_layer(m, math.sqrt(2))
        for m in self.outs:
            init_layer(m, 0.01)
        init_layer(self.glob, math.sqrt(2))
        init_layer(self.done_head, 0.01)
        init_layer(self.type_bias, 0.01)
        init_layer(self.value_head, 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        f, x = self.trunk(obs)
        p = self.proj(f)
        pf = torch.cat([frame(p, k) for k in range(self.nf)], dim=0)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = pf, pf, pf
            else:
                q = F.pad(pf, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        L = torch.cat(res, dim=1).reshape(self.nf, B, self.T, -1)
        w = torch.softmax(SCALE * self.frame_logit, dim=1)
        place = torch.einsum('tk,kbtn->btn', w, L).reshape(B, -1)
        g = F.relu(self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)))
        place = place + self.type_bias(g).repeat_interleave(self.Y * self.X, dim=1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
