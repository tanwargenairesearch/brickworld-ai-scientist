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
PROJ = 16
HID = 12
SCALE = 2.0


def frame(m, k):
    if k >= 4:
        m = m.transpose(-1, -2)
    if k % 2 == 1:
        m = m.flip(-1)
    if (k // 2) % 2 == 1:
        m = m.flip(-2)
    return m


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
        self.nf = 8 if spec.Y == spec.X else 4
        self.stem1 = nn.Conv2d(c, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, COL, 1)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.proj = nn.Conv2d(WIDTH + SKIP, PROJ, 1)
        self.glob = nn.Linear(2 * WIDTH, GH)
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)
        self.type_bias = nn.Linear(GH, self.T)
        for mod in self.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                init_layer(mod, math.sqrt(2))
        init_layer(self.done_head, 0.01)
        init_layer(self.type_bias, 0.01)
        init_layer(self.value_head, 1.0)
        w1 = torch.empty(self.nf, self.T, 3 * PROJ, HID)
        for i in range(self.nf):
            for t in range(self.T):
                nn.init.orthogonal_(w1[i, t], math.sqrt(2))
        self.w1 = nn.Parameter(w1)
        self.b1 = nn.Parameter(torch.zeros(self.nf, self.T, HID))
        w2 = torch.empty(self.nf * self.T, HID)
        nn.init.orthogonal_(w2, 0.01)
        self.w2 = nn.Parameter(w2.reshape(self.nf, self.T, HID))
        self.b2 = nn.Parameter(torch.zeros(self.nf, self.T))
        self.alpha = nn.Parameter(torch.zeros(self.T, self.nf))

    def forward(self, obs, mask):
        s = F.relu(self.stem2(F.relu(self.stem1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        feat = torch.cat([f, F.relu(self.skip(obs))], 1)
        g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)))
        p = self.proj(feat)
        pf = torch.stack([frame(p, k) for k in range(self.nf)], 1)
        B, NF, P, Y, X = pf.shape
        q0 = pf.reshape(B * NF, P, Y, X)
        outs = []
        for t, (dy, dx) in enumerate(self.fps):
            if dy * dx == 1:
                z = torch.cat([q0, q0, q0], 1)
            else:
                q = F.pad(q0, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
                z = torch.cat([mx, mn, av], 1)
            z = z.reshape(B, NF, 3 * P, Y, X)
            h = F.relu(torch.einsum('bfcyx,fch->bfhyx', z, self.w1[:, t]) + self.b1[:, t, :, None, None])
            o = torch.einsum('bfhyx,fh->bfyx', h, self.w2[:, t]) + self.b2[:, t, None, None]
            outs.append(o)
        L = torch.stack(outs, 2)
        L = L + self.type_bias(g)[:, None, :, None, None]
        L = L.reshape(B, NF, self.T, Y * X)
        w = torch.softmax(SCALE * self.alpha, 1)
        place = torch.einsum('bftn,tf->btn', L, w).flatten(1)
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
