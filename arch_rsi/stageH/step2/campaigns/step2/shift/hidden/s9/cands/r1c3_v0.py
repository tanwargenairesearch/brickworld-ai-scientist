import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
LAND = 16
HID = 32
HEADS = 4

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 3, "gamma": 0.97, "lr": 0.001}


class FootHead(nn.Module):
    def __init__(self, c, footprints, land, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        n = len(self.fps)
        self.land = nn.ModuleList([nn.Conv2d(c, land, 1) for _ in range(n)])
        self.landp = nn.ModuleList([nn.Conv2d(2 * land, hid, 1) for _ in range(n)])
        self.cellp = nn.ModuleList([nn.Conv2d(c, hid, 1) for _ in range(n)])
        self.mix = nn.ModuleList([nn.Conv2d(3 * hid, hid, 1) for _ in range(n)])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in range(n)])

    def forward(self, f):
        Y = f.shape[2]
        X = f.shape[3]
        res = []
        for k, (dy, dx) in enumerate(self.fps):
            l = self.land[k](f)
            q = F.pad(l, (0, dx - 1, 0, dy - 1))
            lmax = F.max_pool2d(q, (dy, dx), stride=1)
            lavg = F.avg_pool2d(q, (dy, dx), stride=1)
            a = self.landp[k](torch.cat([lmax, lavg], dim=1))
            c = self.cellp[k](f)
            cp = F.pad(c, (0, dx - 1, 0, dy - 1))
            hs = []
            for i in range(dy):
                for j in range(dx):
                    hs.append(F.relu(cp[:, :, i:i + Y, j:j + X] + a))
            h = torch.stack(hs, dim=0)
            z = torch.cat([h.amax(0), h.amin(0), h.mean(0)], dim=1)
            z = F.relu(self.mix[k](z))
            res.append(self.outs[k](z))
        return torch.cat(res, dim=1).flatten(1)


class CellAttn(nn.Module):
    def __init__(self, c, heads):
        super().__init__()
        self.h = heads
        self.d = c // heads
        self.pos = nn.Linear(2, c)
        self.ln1 = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * self.h * self.d)
        self.o = nn.Linear(self.h * self.d, c)
        self.ln2 = nn.LayerNorm(c)
        self.ff1 = nn.Linear(c, 2 * c)
        self.ff2 = nn.Linear(2 * c, c)

    def forward(self, f):
        B, C, Y, X = f.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=f.device, dtype=f.dtype)
        xs = torch.linspace(-1.0, 1.0, X, device=f.device, dtype=f.dtype)
        gy = ys.view(Y, 1).expand(Y, X)
        gx = xs.view(1, X).expand(Y, X)
        coords = torch.stack([gy, gx], dim=-1).reshape(Y * X, 2)
        t = f.flatten(2).transpose(1, 2) + self.pos(coords).unsqueeze(0)
        n = Y * X
        u = self.ln1(t)
        qkv = self.qkv(u).view(B, n, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d), dim=-1)
        y = torch.matmul(att, v).transpose(1, 2).reshape(B, n, self.h * self.d)
        t = t + self.o(y)
        t = t + self.ff2(F.relu(self.ff1(self.ln2(t))))
        return t.transpose(1, 2).reshape(B, C, Y, X)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.attn = CellAttn(WIDTH, HEADS)
        self.head = FootHead(WIDTH, spec.footprints, LAND, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        f = F.relu(self.attn(f))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
