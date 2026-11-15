import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "n_minibatches": 16, "ent_coef": 0.02, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
LD = 32
LE = 16
LH = 16


def remap_x(s, dx):
    X = s.shape[-1]
    v = torch.flip(s[..., :X - dx + 1], dims=[-1])
    if dx > 1:
        v = F.pad(v, (0, dx - 1))
    return v


class LayerHead(nn.Module):
    def __init__(self, in_ch, width, Z, footprints):
        super().__init__()
        self.Z = int(Z)
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.obs_proj = nn.Conv2d(in_ch, LD, 1)
        self.f_proj = nn.Conv2d(width, LD, 1, bias=False)
        self.zemb = nn.Embedding(self.Z, LD)
        self.l2 = nn.Conv2d(LD, LE, 1)
        self.mix = nn.ModuleList([nn.Conv2d(2 * LE, LH, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(LH, 1, 1) for _ in self.fps])

    def forward(self, obs, f):
        a = self.obs_proj(obs) + self.f_proj(f)
        B, d, Y, X = a.shape
        Z = self.Z
        ez = self.zemb.weight.view(1, Z, d, 1, 1)
        h = F.relu(a.unsqueeze(1) + ez).reshape(B * Z, d, Y, X)
        u = F.relu(self.l2(h))
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn = u, u
            else:
                q = F.pad(u, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            s = out(F.relu(mix(torch.cat([mx, mn], dim=1))))
            s = s.view(B, Z, Y, X)
            s = torch.logsumexp(s, dim=1, keepdim=True)
            res.append(remap_x(s, dx))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = LayerHead(spec.in_channels, WIDTH, spec.Z, spec.footprints)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(obs, f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
