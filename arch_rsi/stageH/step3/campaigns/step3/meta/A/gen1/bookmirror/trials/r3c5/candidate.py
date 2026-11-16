import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0015, "n_envs": 256, "rollout_len": 32, "n_minibatches": 64, "ppo_epochs": 1, "ent_coef": 0.01, "gamma": 0.97, "gae_lambda": 0.9}

STEM = 64
W = 48
SK = 16
P = 32
HID = 48
K = 2
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


def below_any(m):
    c = torch.cummax(m, dim=2)[0]
    return torch.cat([torch.zeros_like(m[:, :, :1]), c[:, :, :-1]], dim=2)


def pool5(t, dy, dx, kind):
    b, k, z, y, x = t.shape
    return wpool(t.reshape(b, k * z, y, x), dy, dx, kind).reshape(b, k, z, y, x)


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
        self.Z = Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        G = 3 * K
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.GroupNorm(8, STEM), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.GroupNorm(8, W), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.n1 = nn.GroupNorm(8, W)
        self.skip = nn.Conv2d(cin, SK, 1)
        self.hmap = nn.Conv2d(cin, K, 1)
        self.lmap = nn.Conv2d(K, K * Z, 1, groups=K)
        self.mmap = nn.Conv2d(cin, K * Z, 1)
        self.proj = nn.Conv2d(W + SK, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P + G + K * Z, HID, 1) for f in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for f in self.fps])
        self.direct = nn.ModuleList([nn.Conv2d(G, 1, 1) for f in self.fps])
        self.anchor = nn.Conv2d(W + SK, 1, 1)
        self.type_bias = nn.Parameter(torch.zeros(len(self.fps)))
        self.glob = nn.Sequential(nn.Linear(2 * (W + SK) + K * Z + K, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.done_direct = nn.Linear(K * Z + K, 1)
        self.value_head = nn.Linear(64, 1)
        ortho_init(self, math.sqrt(2.0))
        small_init(self.hmap, 1.0)
        small_init(self.mmap, 1.0)
        nn.init.normal_(self.lmap.weight, 0.0, 1.0)
        nn.init.zeros_(self.lmap.bias)
        for o in self.outs:
            small_init(o, 0.01)
        for d in self.direct:
            zero_init(d)
        small_init(self.anchor, 0.01)
        small_init(self.done_head, 0.01)
        zero_init(self.done_direct)
        small_init(self.value_head, 1.0)

    def forward(self, obs, mask):
        Z = self.Z
        b, y, x = obs.shape[0], obs.shape[2], obs.shape[3]
        x0 = add_coords(obs)
        s = self.stem(x0)
        f = F.relu(s + self.n1(self.c1(s)))
        sk = F.relu(self.skip(x0))
        fs = torch.cat([f, sk], dim=1)
        hm = self.hmap(x0)
        miss = torch.sigmoid(MS * self.mmap(x0)).view(b, K, Z, y, x)
        mb = below_any(miss)
        p = self.proj(fs)
        a = self.anchor(fs)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            lh = wpool(hm, dy, dx, 0)
            lg = torch.softmax(self.lmap(MS * lh).view(b, K, Z, y, x), dim=2)
            fill = (lg * pool5(miss, dy, dx, 1)).sum(2)
            part = (lg * pool5(miss, dy, dx, 2)).sum(2)
            seal = (lg * pool5(mb, dy, dx, 0)).sum(2)
            gf = torch.cat([fill, part, seal], dim=1)
            z = torch.cat([wpool(p, dy, dx, 0), wpool(p, dy, dx, 1), wpool(p, dy, dx, 2), gf, lg.reshape(b, K * Z, y, x)], dim=1)
            hh = F.relu(self.mix[i](z))
            res.append(self.outs[i](hh) + self.direct[i](GS * gf) + a + self.type_bias[i])
        place = torch.cat(res, dim=1).flatten(1)
        ms = torch.cat([miss.flatten(3).amax(3).flatten(1), hm.mean((2, 3))], dim=1)
        g = self.glob(torch.cat([fs.mean((2, 3)), fs.amax((2, 3)), ms], dim=1))
        done = self.done_head(g) + self.done_direct(GS * ms)
        logits = torch.cat([place, done], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
