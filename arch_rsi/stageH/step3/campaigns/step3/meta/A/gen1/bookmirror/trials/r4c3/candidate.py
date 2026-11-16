import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.003, "n_envs": 256, "rollout_len": 32, "n_minibatches": 64, "ppo_epochs": 1, "ent_coef": 0.0, "gamma": 0.995, "gae_lambda": 0.97, "clip_eps": 0.3}

STEM = 64
W = 48
SK = 16
P = 32
HID = 32
MS = 3.0
GS = 4.0


def add_coords(obs):
    b, c, y, x = obs.shape
    ys = torch.linspace(-1.0, 1.0, y, device=obs.device, dtype=obs.dtype).view(1, 1, y, 1).expand(b, 1, y, x)
    xs = torch.linspace(-1.0, 1.0, x, device=obs.device, dtype=obs.dtype).view(1, 1, 1, x).expand(b, 1, y, x)
    return torch.cat([obs, ys, xs], dim=1)


def wpool(t, dy, dx, kind):
    if dy * dx == 1:
        return t
    q = F.pad(t, (0, dx - 1, 0, dy - 1))
    if kind == 0:
        return F.max_pool2d(q, (dy, dx), stride=1)
    if kind == 1:
        return -F.max_pool2d(-q, (dy, dx), stride=1)
    return F.avg_pool2d(q, (dy, dx), stride=1)


def ortho_init(net, gain):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def small_init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels + 2
        Z = spec.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.GroupNorm(8, STEM), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.GroupNorm(8, W), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.n1 = nn.GroupNorm(8, W)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.n2 = nn.GroupNorm(8, W)
        self.skip = nn.Conv2d(cin, SK, 1)
        self.mmap = nn.Conv2d(cin, Z, 1)
        self.proj = nn.Conv2d(W + SK, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, HID, 1) for f in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for f in self.fps])
        self.anchor = nn.Conv2d(W + SK, 1, 1)
        self.type_bias = nn.Parameter(torch.zeros(len(self.fps)))
        self.ltemp = nn.Parameter(torch.zeros(1))
        self.glob = nn.Sequential(nn.Linear(2 * (W + SK) + Z, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.done_direct = nn.Linear(Z, 1)
        self.value_head = nn.Linear(64, 1)
        ortho_init(self, math.sqrt(2.0))
        small_init(self.mmap, 1.0)
        for o in self.outs:
            small_init(o, 0.01)
        small_init(self.anchor, 0.01)
        small_init(self.done_head, 0.01)
        zero_init(self.done_direct)
        small_init(self.value_head, 1.0)

    def forward(self, obs, mask):
        x0 = add_coords(obs)
        s = self.stem(x0)
        h = F.relu(self.n1(self.c1(s)))
        f = F.relu(s + self.n2(self.c2(h)))
        sk = F.relu(self.skip(x0))
        fs = torch.cat([f, sk], dim=1)
        p = self.proj(fs)
        a = self.anchor(fs)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            z = torch.cat([wpool(p, dy, dx, 0), wpool(p, dy, dx, 1), wpool(p, dy, dx, 2)], dim=1)
            hh = F.relu(self.mix[i](z))
            res.append(self.outs[i](hh) + a + self.type_bias[i])
        place = torch.cat(res, dim=1).flatten(1)
        ms = torch.sigmoid(MS * self.mmap(x0)).amax((2, 3))
        g = self.glob(torch.cat([fs.mean((2, 3)), fs.amax((2, 3)), ms], dim=1))
        done = self.done_head(g) + self.done_direct(GS * ms)
        logits = torch.cat([place, done], dim=1) * torch.exp(self.ltemp.clamp(-2.0, 3.0))
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
