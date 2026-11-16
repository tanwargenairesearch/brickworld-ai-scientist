import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 64, "rollout_len": 64, "n_minibatches": 16, "ppo_epochs": 1, "ent_coef": 0.02, "gamma": 0.995, "gae_lambda": 0.99, "clip_eps": 0.3}

STEM = 64
W = 48
SK = 16
P = 12
HID = 32


def add_coords(obs):
    b, c, y, x = obs.shape
    ys = torch.linspace(-1.0, 1.0, y, device=obs.device, dtype=obs.dtype).view(1, 1, y, 1).expand(b, 1, y, x)
    xs = torch.linspace(-1.0, 1.0, x, device=obs.device, dtype=obs.dtype).view(1, 1, 1, x).expand(b, 1, y, x)
    return torch.cat([obs, ys, xs], dim=1)


def shift(t, dy, dx, pad):
    y, x = t.shape[2], t.shape[3]
    q = F.pad(t, (pad, pad, pad, pad))
    return q[:, :, pad + dy:pad + dy + y, pad + dx:pad + dx + x]


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


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels + 2
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.pad = max([max(a, b) for a, b in self.fps]) + 1
        self.offs = [[(r, c) for r in range(-1, dy + 1) for c in range(-1, dx + 1)] for dy, dx in self.fps]
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.GroupNorm(8, STEM), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.GroupNorm(8, W), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.n1 = nn.GroupNorm(8, W)
        self.skip = nn.Conv2d(cin, SK, 1)
        self.proj = nn.Conv2d(W + SK, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(len(o) * P, HID, 1) for o in self.offs])
        self.mix2 = nn.ModuleList([nn.Conv2d(HID, HID, 1) for o in self.offs])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for o in self.offs])
        self.anchor = nn.Conv2d(W + SK, 1, 1)
        self.type_bias = nn.Parameter(torch.zeros(len(self.fps)))
        self.glob = nn.Sequential(nn.Linear(2 * (W + SK), 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        ortho_init(self, math.sqrt(2.0))
        for o in self.outs:
            small_init(o, 0.01)
        small_init(self.anchor, 0.01)
        small_init(self.done_head, 0.01)
        small_init(self.value_head, 1.0)

    def forward(self, obs, mask):
        x0 = add_coords(obs)
        s = self.stem(x0)
        f = F.relu(s + self.n1(self.c1(s)))
        sk = F.relu(self.skip(x0))
        fs = torch.cat([f, sk], dim=1)
        p = self.proj(fs)
        a = self.anchor(fs)
        res = []
        for i in range(len(self.fps)):
            parts = [shift(p, r, c, self.pad) for r, c in self.offs[i]]
            hh = F.relu(self.mix[i](torch.cat(parts, dim=1)))
            hh = F.relu(self.mix2[i](hh))
            res.append(self.outs[i](hh) + a + self.type_bias[i])
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([fs.mean((2, 3)), fs.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
