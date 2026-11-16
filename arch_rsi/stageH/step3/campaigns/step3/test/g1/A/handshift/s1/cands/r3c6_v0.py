import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 8, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.008, "gamma": 0.97, "gae_lambda": 0.92, "clip_eps": 0.2}

STEM = 48
W = 40
PROJ = 20
RAW = 12
HID = 32
GLOB = 64


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain=gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def zero_layer(m):
    nn.init.zeros_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def add_coords(obs):
    b, c, y, x = obs.shape
    ys = torch.linspace(-1.0, 1.0, y, device=obs.device, dtype=obs.dtype).view(1, 1, y, 1).expand(b, 1, y, x)
    xs = torch.linspace(-1.0, 1.0, x, device=obs.device, dtype=obs.dtype).view(1, 1, 1, x).expand(b, 1, y, x)
    return torch.cat([obs, ys, xs], dim=1)


def fp_spread(p, dy, dx):
    if dy * dx == 1:
        return torch.cat([p, p, p, torch.zeros_like(p)], dim=1)
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return torch.cat([mx, mn, av, mx - mn], dim=1)


def done_stats(maps, fps):
    vals = []
    for m, (dy, dx) in zip(maps, fps):
        y, x = m.shape[2], m.shape[3]
        vals.append(m[:, 0, :max(1, y - dy + 1), :max(1, x - dx + 1)].flatten(1))
    v = torch.cat(vals, dim=1).detach()
    lse = torch.logsumexp(v, dim=1) - math.log(v.shape[1])
    return torch.stack([v.amax(1), lse, v.mean(1)], dim=1)


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = init_layer(nn.Conv2d(c, c, 3, padding=1))
        self.n1 = nn.GroupNorm(8, c)
        self.c2 = init_layer(nn.Conv2d(c, c, 3, padding=1))
        self.n2 = nn.GroupNorm(8, c)

    def forward(self, x):
        h = F.silu(self.n1(self.c1(x)))
        return F.silu(x + self.n2(self.c2(h)))


class SharedHead(nn.Module):
    def __init__(self, cat, hid, n_types):
        super().__init__()
        self.l1 = init_layer(nn.Conv2d(4 * cat, hid, 1))
        self.emb = nn.Parameter(torch.zeros(n_types, hid))
        self.l2 = init_layer(nn.Conv2d(hid, hid, 1))
        self.l3 = init_layer(nn.Conv2d(hid, 1, 1), 0.01)

    def forward(self, p, fps):
        maps = []
        for i, (dy, dx) in enumerate(fps):
            h = F.silu(self.l1(fp_spread(p, dy, dx)) + self.emb[i].view(1, -1, 1, 1))
            h = F.silu(self.l2(h))
            maps.append(self.l3(h))
        return maps


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels + 2
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            init_layer(nn.Conv2d(cin, STEM, 1)), nn.SiLU(),
            init_layer(nn.Conv2d(STEM, W, 1)), nn.GroupNorm(8, W), nn.SiLU())
        self.c0 = init_layer(nn.Conv2d(W, W, 3, padding=1))
        self.n0 = nn.GroupNorm(8, W)
        self.res = Res(W)
        self.proj = init_layer(nn.Conv2d(2 * W, PROJ, 1), 1.0)
        self.raw = init_layer(nn.Conv2d(cin, RAW, 1), 1.0)
        cat = PROJ + RAW
        self.head = SharedHead(cat, HID, len(self.fps))
        self.anchor = init_layer(nn.Conv2d(cat, 1, 1), 0.01)
        self.tbias = nn.Parameter(torch.zeros(len(self.fps)))
        self.glob = nn.Sequential(init_layer(nn.Linear(4 * W, GLOB)), nn.SiLU())
        self.done_head = init_layer(nn.Linear(GLOB, 1), 0.01)
        self.done_stat = zero_layer(nn.Linear(3, 1))
        self.value_head = init_layer(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        x = add_coords(obs)
        s = self.stem(x)
        h = F.silu(s + self.n0(self.c0(s)))
        f = self.res(h)
        p = torch.cat([self.proj(torch.cat([f, s], dim=1)), self.raw(x)], dim=1)
        a = self.anchor(p)
        raw_maps = self.head(p, self.fps)
        maps = [m + a + self.tbias[i] for i, m in enumerate(raw_maps)]
        place = torch.cat(maps, dim=1).flatten(1)
        gf = torch.cat([f.mean((2, 3)), f.amax((2, 3)), s.mean((2, 3)), s.amax((2, 3))], dim=1)
        gh = self.glob(gf)
        done = self.done_head(gh) + self.done_stat(done_stats(maps, self.fps))
        logits = torch.cat([place, done], dim=1)
        return logits, self.value_head(gh).squeeze(1)


def build_policy(spec):
    return Net(spec)
