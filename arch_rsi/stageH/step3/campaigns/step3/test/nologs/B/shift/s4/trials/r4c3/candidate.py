import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.0007, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.003}

STEM = 96
D = 48
NH = 4
FF = 96
NL = 2
HID = 24


def small_init(m):
    nn.init.normal_(m.weight, std=0.01)
    nn.init.zeros_(m.bias)


class TBlock(nn.Module):
    def __init__(self, d, nh, ff):
        super().__init__()
        self.nh = nh
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, ff)
        self.f2 = nn.Linear(ff, d)

    def forward(self, x):
        B, N, Dm = x.shape
        hd = Dm // self.nh
        h = self.n1(x)
        qkv = self.qkv(h).view(B, N, 3, self.nh, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd), dim=-1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(B, N, Dm)
        x = x + self.o(y)
        return x + self.f2(F.relu(self.f1(self.n2(x))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y, X = int(spec.Y), int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, D, 1))
        self.pos = nn.Parameter(torch.zeros(1, D, Y, X))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([TBlock(D, NH, FF) for _ in range(NL)])
        self.nf = nn.LayerNorm(D)
        self.convs = nn.ModuleList([nn.Conv2d(D, HID, (dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        for o in self.outs:
            small_init(o)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        small_init(self.done_head)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        f = self.stem(obs) + self.pos
        t = f.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            t = blk(t)
        t = self.nf(t)
        f = t.transpose(1, 2).reshape(B, D, Y, X)
        res = []
        for (dy, dx), conv, out in zip(self.fps, self.convs, self.outs):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            res.append(out(F.relu(conv(q))))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
