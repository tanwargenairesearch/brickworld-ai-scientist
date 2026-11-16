import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.97, "gae_lambda": 0.9, "clip_eps": 0.25}

STEM = 48
W = 32
NBLK = 2
PROJ = 20
RAW = 12
HID = 32
GRAW = 16
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


def fp_cells(p, dy, dx):
    if dy * dx == 1:
        return p
    y, x = p.shape[2], p.shape[3]
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    parts = [q[:, :, i:i + y, j:j + x] for i in range(dy) for j in range(dx)]
    parts.append(F.max_pool2d(q, (dy, dx), stride=1))
    parts.append(-F.max_pool2d(-q, (dy, dx), stride=1))
    return torch.cat(parts, dim=1)


def done_stats(maps, fps):
    vals = []
    for m, (dy, dx) in zip(maps, fps):
        y, x = m.shape[2], m.shape[3]
        vals.append(m[:, 0, :max(1, y - dy + 1), :max(1, x - dx + 1)].flatten(1))
    v = torch.cat(vals, dim=1).detach()
    lse = torch.logsumexp(v, dim=1) - math.log(v.shape[1])
    return torch.stack([v.amax(1), lse, v.mean(1)], dim=1)


class GLURes(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = init_layer(nn.Conv2d(c, 2 * c, 3, padding=1), 1.0)
        self.n = nn.GroupNorm(8, c)

    def forward(self, x):
        u, v = self.conv(x).chunk(2, dim=1)
        return x + self.n(u * torch.sigmoid(v))


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
        self.stem1 = init_layer(nn.Conv2d(cin, 2 * STEM, 1), 1.0)
        self.stem2 = init_layer(nn.Conv2d(STEM, W, 1))
        self.ns = nn.GroupNorm(8, W)
        self.blocks = nn.ModuleList([GLURes(W) for _ in range(NBLK)])
        self.proj = init_layer(nn.Conv2d(2 * W, PROJ, 1), 1.0)
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
        self.abias = nn.Parameter(torch.zeros(spec.n_actions))
        self.gproj = init_layer(nn.Conv2d(cin, GRAW, 1))
        self.glob = nn.Sequential(init_layer(nn.Linear(4 * W + 2 * GRAW, GLOB)), nn.ReLU())
        self.done_head = init_layer(nn.Linear(GLOB, 1), 0.01)
        self.done_stat = zero_layer(nn.Linear(3, 1))
        self.value_head = init_layer(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        x = add_coords(obs)
        u, v = self.stem1(x).chunk(2, dim=1)
        s = F.relu(self.ns(self.stem2(u * torch.sigmoid(v))))
        f = s
        for blk in self.blocks:
            f = blk(f)
        f = F.relu(f)
        p = torch.cat([self.proj(torch.cat([f, s], dim=1)), self.raw(x)], dim=1)
        a = self.anchor(p)
        maps = []
        for i, (dy, dx) in enumerate(self.fps):
            maps.append(self.heads[i](fp_cells(p, dy, dx)) + a + self.tbias[i])
        place = torch.cat(maps, dim=1).flatten(1)
        g = F.relu(self.gproj(x))
        gf = torch.cat([f.mean((2, 3)), f.amax((2, 3)), s.mean((2, 3)), s.amax((2, 3)), g.mean((2, 3)), g.amax((2, 3))], dim=1)
        gh = self.glob(gf)
        done = self.done_head(gh) + self.done_stat(done_stats(maps, self.fps))
        logits = torch.cat([place, done], dim=1) + self.abias
        return logits, self.value_head(gh).squeeze(1)


def build_policy(spec):
    return Net(spec)
