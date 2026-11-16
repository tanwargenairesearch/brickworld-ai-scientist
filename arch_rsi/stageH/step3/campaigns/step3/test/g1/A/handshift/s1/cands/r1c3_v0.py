import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0005, "n_envs": 64, "rollout_len": 32, "n_minibatches": 8, "ppo_epochs": 3, "ent_coef": 0.0, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}

STEM = 64
W = 40
PROJ = 24
RAW = 16
HID = 32
GRAW = 16
GLOB = 64


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain=gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def add_coords(obs):
    b, c, y, x = obs.shape
    ys = torch.linspace(-1.0, 1.0, y, device=obs.device, dtype=obs.dtype).view(1, 1, y, 1).expand(b, 1, y, x)
    xs = torch.linspace(-1.0, 1.0, x, device=obs.device, dtype=obs.dtype).view(1, 1, 1, x).expand(b, 1, y, x)
    return torch.cat([obs, ys, xs], dim=1)


def fp_cells(p, dy, dx):
    if dy * dx == 1:
        return p
    y, x = p.shape[2], p.shape[3]
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    parts = [q[:, :, i:i + y, j:j + x] for i in range(dy) for j in range(dx)]
    parts.append(F.max_pool2d(q, (dy, dx), stride=1))
    parts.append(-F.max_pool2d(-q, (dy, dx), stride=1))
    return torch.cat(parts, dim=1)


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = init_layer(nn.Conv2d(c, c, 3, padding=1))
        self.n1 = nn.GroupNorm(8, c)
        self.c2 = init_layer(nn.Conv2d(c, c, 3, padding=1))
        self.n2 = nn.GroupNorm(8, c)

    def forward(self, x):
        h = F.relu(self.n1(self.c1(x)))
        return F.relu(x + self.n2(self.c2(h)))


class TypeHead(nn.Module):
    def __init__(self, cin, hid):
        super().__init__()
        self.l1 = init_layer(nn.Conv2d(cin, hid, 1))
        self.l2 = init_layer(nn.Conv2d(hid, hid, 1))
        self.l3 = init_layer(nn.Conv2d(hid, 1, 1), 0.01)

    def forward(self, z):
        return self.l3(F.relu(self.l2(F.relu(self.l1(z)))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels + 2
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            init_layer(nn.Conv2d(cin, STEM, 1)), nn.ReLU(),
            init_layer(nn.Conv2d(STEM, W, 1)), nn.GroupNorm(8, W), nn.ReLU())
        self.c0 = init_layer(nn.Conv2d(W, W, 3, padding=1))
        self.n0 = nn.GroupNorm(8, W)
        self.res = Res(W)
        self.proj = init_layer(nn.Conv2d(W, PROJ, 1), 1.0)
        self.raw = init_layer(nn.Conv2d(cin, RAW, 1), 1.0)
        cat = PROJ + RAW
        heads = []
        for dy, dx in self.fps:
            n = dy * dx
            k = cat if n == 1 else (n + 2) * cat
            heads.append(TypeHead(k, HID))
        self.heads = nn.ModuleList(heads)
        self.anchor = init_layer(nn.Conv2d(cat, 1, 1), 0.01)
        self.tbias = nn.Parameter(torch.zeros(len(self.fps)))
        self.gproj = init_layer(nn.Conv2d(cin, GRAW, 1))
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * W + 2 * GRAW, GLOB)), nn.ReLU())
        self.done_head = init_layer(nn.Linear(GLOB, 1), 0.01)
        self.value_head = init_layer(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        x = add_coords(obs)
        s = self.stem(x)
        h = F.relu(self.n0(self.c0(s)))
        f = self.res(h)
        p = torch.cat([self.proj(f), self.raw(x)], dim=1)
        a = self.anchor(p)
        outs = []
        for i, (dy, dx) in enumerate(self.fps):
            outs.append(self.heads[i](fp_cells(p, dy, dx)) + a + self.tbias[i])
        place = torch.cat(outs, dim=1).flatten(1)
        g = F.relu(self.gproj(x))
        gf = torch.cat([f.mean((2, 3)), f.amax((2, 3)), g.mean((2, 3)), g.amax((2, 3))], dim=1)
        gh = self.glob(gf)
        logits = torch.cat([place, self.done_head(gh)], dim=1)
        return logits, self.value_head(gh).squeeze(1)


def build_policy(spec):
    return Net(spec)
