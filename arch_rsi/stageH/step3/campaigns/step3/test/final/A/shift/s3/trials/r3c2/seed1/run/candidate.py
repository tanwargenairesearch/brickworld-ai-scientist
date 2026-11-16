import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 48
WIDTH = 32
SKIP = 16
HID = 24
MIX_SCALE = 5.0


def frame_apply(m, k, square):
    d0 = m.dim() - 2
    d1 = m.dim() - 1
    if square:
        r = k % 4
        if r > 0:
            m = torch.rot90(m, r, dims=(d0, d1))
        if k >= 4:
            m = m.transpose(d0, d1)
    else:
        if k % 2 == 1:
            m = m.flip(d1)
        if (k // 2) % 2 == 1:
            m = m.flip(d0)
    return m


def ortho_init(net):
    for mod in net.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(mod.weight, math.sqrt(2.0))
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)


def set_gain(mod, gain):
    nn.init.orthogonal_(mod.weight, gain)
    nn.init.zeros_(mod.bias)


class Trunk(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(c, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, STEM, 1), nn.ReLU())
        self.c1 = nn.Conv2d(STEM, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        k = F.relu(self.skip(obs))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        return torch.cat([f, k], dim=1), g


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.C = int(spec.in_channels)
        self.square = self.Y == self.X
        self.nf = 8 if self.square else 4
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(fps)
        K = max(max(a, b) for a, b in fps)
        win = 2 * K - 1
        self.trunk = Trunk(self.nf * self.C)
        self.hconv = nn.Conv2d(WIDTH + SKIP, HID, win, padding=K - 1)
        self.out = nn.Conv2d(HID, self.nf * self.T, 1)
        self.dense = nn.Linear(64, int(spec.n_place))
        self.type_bias = nn.Linear(64, self.T)
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        ortho_init(self)
        set_gain(self.out, 0.01)
        set_gain(self.dense, 0.01)
        set_gain(self.type_bias, 0.01)
        set_gain(self.done_head, 0.01)
        set_gain(self.value_head, 1.0)
        self.mix = nn.Parameter(torch.zeros(2 * self.nf))

    def forward(self, obs, mask):
        views = torch.cat([frame_apply(obs, k, self.square) for k in range(self.nf)], dim=1)
        h, g = self.trunk(views)
        B = h.shape[0]
        s = self.out(F.relu(self.hconv(h)))
        s = s.reshape(B, self.nf, self.T, self.Y, self.X) + self.type_bias(g)[:, None, :, None, None]
        comps = []
        for k in range(self.nf):
            m = frame_apply(s[:, k], k, self.square)
            comps.append(m.reshape(B, -1))
            comps.append(m.permute(0, 2, 3, 1).reshape(B, -1))
        st = torch.stack(comps, dim=1)
        w = torch.softmax(MIX_SCALE * self.mix, dim=0)
        place = torch.einsum('bkn,k->bn', st, w) + self.dense(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
