import math
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
HEADS = 4
HID = 24


class FootHead(nn.Module):
    def __init__(self, c, footprints, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for (a, b) in footprints]
        self.pre = nn.Conv2d(c, hid, 1)
        self.mlps = nn.ModuleList()
        for _ in self.fps:
            out = nn.Conv2d(hid, 1, 1)
            nn.init.normal_(out.weight, std=0.01)
            nn.init.zeros_(out.bias)
            self.mlps.append(nn.Sequential(nn.Conv2d(3 * hid, hid, 1), nn.ReLU(), out))

    def forward(self, f):
        h = self.pre(f)
        outs = []
        for fp, mlp in zip(self.fps, self.mlps):
            dy, dx = fp
            hp = F.pad(h, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(hp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-hp, (dy, dx), stride=1)
            av = F.avg_pool2d(hp, (dy, dx), stride=1)
            outs.append(mlp(torch.cat([mx, mn, av], 1)))
        return torch.cat(outs, 1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.inp = nn.Conv2d(spec.in_channels + 2, WIDTH, 3, padding=1)
        self.ln = nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.proj = nn.Linear(WIDTH, WIDTH)
        self.conv = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.place_head = FootHead(WIDTH, spec.footprints, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        b, Y, X = obs.shape[0], obs.shape[2], obs.shape[3]
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(b, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(b, 1, Y, X)
        f = F.relu(self.inp(torch.cat([obs, ys, xs], dim=1)))
        n = Y * X
        d = WIDTH // HEADS
        t = f.flatten(2).transpose(1, 2)
        qkv = self.qkv(self.ln(t)).view(b, n, 3, HEADS, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d), dim=-1)
        a = torch.matmul(att, v).transpose(1, 2).reshape(b, n, WIDTH)
        t = t + self.proj(a)
        f = t.transpose(1, 2).reshape(b, WIDTH, Y, X)
        f = f + F.relu(self.conv(f))
        place = self.place_head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
