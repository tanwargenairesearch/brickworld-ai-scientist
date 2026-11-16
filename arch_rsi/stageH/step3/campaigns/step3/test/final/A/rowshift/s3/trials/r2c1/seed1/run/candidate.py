import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 36
SKIP = 16
WHID = 24
GLOB = 64
SCALE = 5.0
ORDERS = [("t", "y", "x"), ("t", "x", "y"), ("y", "t", "x"), ("x", "t", "y"), ("y", "x", "t"), ("x", "y", "t")]


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def build_options():
    opts = []
    for order in ORDERS:
        for fy in range(2):
            for fx in range(2):
                for dfirst in range(2):
                    opts.append((order, fy, fx, dfirst))
    return opts


def build_index(Y, X, T, n_actions, opts):
    YX = Y * X
    P = T * YX
    sizes = {"t": T, "y": Y, "x": X}
    done_slot = len(opts) * P
    rows = []
    for o, opt in enumerate(opts):
        order, fy, fx, dfirst = opt
        row = []
        for e in range(n_actions):
            a = e - 1 if dfirst == 1 else e
            if a < 0 or a >= P:
                row.append(done_slot)
                continue
            vals = {}
            r = a
            for ax in reversed(order):
                vals[ax] = r % sizes[ax]
                r = r // sizes[ax]
            t = vals["t"]
            y = vals["y"]
            x = vals["x"]
            if fy == 1:
                y = Y - 1 - y
            if fx == 1:
                x = X - 1 - x
            row.append(o * P + t * YX + y * X + x)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long)


class Trunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = int(spec.in_channels)
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


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(fps)
        K = max(max(a, b) for a, b in fps)
        opts = build_options()
        self.nopt = len(opts)
        self.trunk = Trunk(spec)
        self.win = nn.Conv2d(WIDTH + SKIP, WHID, 2 * K - 1, padding=K - 1)
        self.wout = nn.Conv2d(WHID, self.nopt * T, 1)
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        self.frame_logit = nn.Parameter(torch.zeros(self.nopt))
        self.register_buffer("idx", build_index(Y, X, T, int(spec.n_actions), opts))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                ortho(m, math.sqrt(2))
        ortho(self.wout, 0.01)
        ortho(self.done_head, 0.01)
        ortho(self.value_head, 1.0)

    def forward(self, obs, mask):
        h, g = self.trunk(obs)
        m = self.wout(F.relu(self.win(h)))
        flat = m.flatten(1)
        flat = torch.cat([flat, self.done_head(g)], dim=1)
        per = flat[:, self.idx]
        w = torch.softmax(SCALE * self.frame_logit, 0)
        logits = torch.einsum("bon,o->bn", per, w)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
