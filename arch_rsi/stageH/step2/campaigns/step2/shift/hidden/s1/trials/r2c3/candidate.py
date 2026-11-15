import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
D = 48
HEADS = 4
FF = 96
NL = 2
HID = 32
GLOB = 64


class Block(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.h = h
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, ff)
        self.f2 = nn.Linear(ff, d)

    def forward(self, x):
        B, N, Dm = x.shape
        hd = Dm // self.h
        qkv = self.qkv(self.n1(x)).view(B, N, 3, self.h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd), dim=-1)
        y = torch.matmul(att, v).transpose(1, 2).reshape(B, N, Dm)
        x = x + self.o(y)
        return x + self.f2(F.relu(self.f1(self.n2(x))))


class WindowHead(nn.Module):
    def __init__(self, c, footprints, hid, gdim):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.win = nn.ModuleList([nn.Conv2d(c, hid, (dy, dx)) for dy, dx in self.fps])
        self.gin = nn.ModuleList([nn.Linear(gdim, hid) for _ in self.fps])
        self.mix = nn.ModuleList([nn.Conv2d(hid, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f, g):
        res = []
        for (dy, dx), win, gin, mix, out in zip(self.fps, self.win, self.gin, self.mix, self.outs):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            h = F.relu(win(q) + gin(g)[:, :, None, None])
            h = F.relu(mix(h))
            res.append(out(h).flatten(1))
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + 2, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, D, 1), nn.ReLU())
        self.blocks = nn.ModuleList([Block(D, HEADS, FF) for _ in range(NL)])
        self.nf = nn.LayerNorm(D)
        self.glob = nn.Sequential(nn.Linear(2 * D, GLOB), nn.ReLU())
        self.head = WindowHead(2 * D, spec.footprints, HID, GLOB)
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Sequential(nn.Linear(GLOB, GLOB), nn.ReLU(), nn.Linear(GLOB, 1))

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        s = self.stem(torch.cat([obs, yy, xx], dim=1))
        t = s.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            t = blk(t)
        t = F.relu(self.nf(t))
        f = t.transpose(1, 2).reshape(B, D, Y, X)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        place = self.head(torch.cat([s, f], dim=1), g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
