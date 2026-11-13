import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
WIDTH = 32
HEADS = 4
K = 6
HID = 12


def zero_layer(m):
    nn.init.zeros_(m.weight)
    nn.init.zeros_(m.bias)
    return m


class Attn(nn.Module):
    def __init__(self, width, heads, n):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.pos = nn.Parameter(torch.zeros(1, n, width))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x):
        B, C, Y, X = x.shape
        N = Y * X
        H = self.heads
        d = C // H
        t = x.flatten(2).transpose(1, 2)
        u = self.norm(t) + self.pos
        q, k, v = self.qkv(u).chunk(3, dim=-1)
        q = q.reshape(B, N, H, d).transpose(1, 2)
        k = k.reshape(B, N, H, d).transpose(1, 2)
        v = v.reshape(B, N, H, d).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d), dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, N, C)
        t = t + self.out(o)
        return t.transpose(1, 2).reshape(B, C, Y, X)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.Z = spec.Z
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.conv1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.conv3 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.attn = Attn(WIDTH, HEADS, spec.Y * spec.X)
        self.proj = nn.Conv2d(WIDTH, spec.Z * K, 1)
        self.type_hidden = nn.ModuleList([nn.Linear(6 * K, HID) for _ in self.fps])
        self.layer_emb = nn.Parameter(torch.zeros(len(self.fps), spec.Z, HID))
        nn.init.normal_(self.layer_emb, std=0.02)
        self.type_out = nn.ModuleList([zero_layer(nn.Linear(HID, 1)) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = zero_layer(nn.Linear(64, 1))
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        x = self.stem(obs)
        x = x + F.relu(self.conv1(x))
        x = x + F.relu(self.conv2(x))
        x = self.attn(x)
        x = x + F.relu(self.conv3(x))
        B, _, Y, X = x.shape
        Z = self.Z
        h = self.proj(x)
        h5 = h.reshape(B, Z, K, Y, X)
        outs = []
        for i, fp in enumerate(self.fps):
            dy, dx = fp
            hp = F.pad(h, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(hp, (dy, dx), stride=1).reshape(B, Z, K, Y, X)
            mn = (-F.max_pool2d(-hp, (dy, dx), stride=1)).reshape(B, Z, K, Y, X)
            av = F.avg_pool2d(hp, (dy, dx), stride=1).reshape(B, Z, K, Y, X)
            bmx = F.pad(mx, (0, 0, 0, 0, 0, 0, 1, 0))[:, :Z]
            bmn = F.pad(mn, (0, 0, 0, 0, 0, 0, 1, 0))[:, :Z]
            z = torch.cat([h5, mx, mn, av, bmx, bmn], dim=2)
            z = z.permute(0, 1, 3, 4, 2)
            z = self.type_hidden[i](z) + self.layer_emb[i][None, :, None, None, :]
            z = F.relu(z)
            s = self.type_out[i](z).squeeze(-1)
            s = torch.logsumexp(s, dim=1)
            outs.append(s.unsqueeze(1))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
