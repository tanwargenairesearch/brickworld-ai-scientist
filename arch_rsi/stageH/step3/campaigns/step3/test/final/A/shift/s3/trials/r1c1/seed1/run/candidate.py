import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

WIDTH = 36
SKIP = 16
PROJ = 12
HID = 12
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
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.stem = nn.Sequential(nn.Conv2d(c, 64, 1), nn.ReLU(), nn.Conv2d(64, 48, 1), nn.ReLU())
        self.c1 = nn.Conv2d(48, WIDTH, 3, padding=1)
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
        self.square = self.Y == self.X
        self.nf = 8 if self.square else 4
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(self.fps)
        nf, T = self.nf, self.T
        self.trunk = Trunk(spec)
        self.proj = nn.Conv2d(WIDTH + SKIP, PROJ, 1)
        self.type_bias = nn.Linear(64, T)
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        ortho_init(self)
        set_gain(self.type_bias, 0.01)
        set_gain(self.done_head, 0.01)
        set_gain(self.value_head, 1.0)
        self.w1 = nn.Parameter(torch.randn(nf, T, HID, 3 * PROJ) * math.sqrt(2.0 / (3 * PROJ)))
        self.b1 = nn.Parameter(torch.zeros(nf, T, HID))
        self.w2 = nn.Parameter(torch.randn(nf, T, HID) * (0.01 / math.sqrt(HID)))
        self.b2 = nn.Parameter(torch.zeros(nf, T))
        self.mix = nn.Parameter(torch.zeros(nf))

    def forward(self, obs, mask):
        h, g = self.trunk(obs)
        p = F.relu(self.proj(h))
        B = p.shape[0]
        fr = torch.stack([frame_apply(p, k, self.square) for k in range(self.nf)], dim=1)
        Yf = fr.shape[-2]
        Xf = fr.shape[-1]
        q0 = fr.reshape(B * self.nf, PROJ, Yf, Xf)
        feats = []
        for (dy, dx) in self.fps:
            if dy * dx == 1:
                mx, mn, av = q0, q0, q0
            else:
                q = F.pad(q0, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            feats.append(torch.cat([mx, mn, av], dim=1))
        z = torch.stack(feats, dim=1).reshape(B, self.nf, self.T, 3 * PROJ, Yf * Xf)
        hdn = F.relu(torch.einsum('bftcn,fthc->bfthn', z, self.w1) + self.b1[None, :, :, :, None])
        out = torch.einsum('bfthn,fth->bftn', hdn, self.w2) + self.b2[None, :, :, None]
        out = out + self.type_bias(g)[:, None, :, None]
        w = torch.softmax(MIX_SCALE * self.mix, dim=0)
        place = torch.einsum('bftn,f->btn', out, w).reshape(B, -1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
