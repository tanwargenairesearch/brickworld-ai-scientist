import math
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 32
HEADS = 2
HID = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.n_types = spec.n_types
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.pre = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(2)])
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.proj = nn.Linear(WIDTH, WIDTH)
        self.post = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.type_feat = nn.Conv2d(WIDTH, self.n_types * HID, 1)
        self.type_out = nn.ModuleList([nn.Conv2d(3 * HID, 1, 1) for _ in range(self.n_types)])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = self.stem(obs)
        for conv in self.pre:
            f = f + conv(F.relu(f))
        B, W, Y, X = f.shape
        N = Y * X
        d = W // HEADS
        tok = F.relu(f).flatten(2).transpose(1, 2)
        qkv = self.qkv(tok).view(B, N, 3, HEADS, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d), dim=-1)
        o = torch.matmul(att, v).transpose(1, 2).reshape(B, N, W)
        f = f + self.proj(o).transpose(1, 2).reshape(B, W, Y, X)
        f = f + self.post(F.relu(f))
        f = F.relu(f)
        h = self.type_feat(f)
        outs = []
        for t in range(self.n_types):
            dy, dx = self.footprints[t]
            ht = h[:, t * HID:(t + 1) * HID]
            hp = F.pad(ht, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(hp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-hp, (dy, dx), stride=1)
            mean = F.avg_pool2d(hp, (dy, dx), stride=1)
            outs.append(self.type_out[t](torch.cat([mx, mn, mean], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
