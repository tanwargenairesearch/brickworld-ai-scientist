"""r3c3 recipe + conv body + one global self-attention block over cells + context + footprint heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
HID = 32
NHEAD = 4
RECIPE = {"gamma": 0.99, "gae_lambda": 0.95, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2}


def fp_pool(f, dy, dx):
    p = F.pad(f, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(p, (dy, dx), stride=1)
    mn = -F.max_pool2d(-p, (dy, dx), stride=1)
    return mx, mn


class Attn(nn.Module):
    def __init__(self, dim, nhead):
        super().__init__()
        self.nhead = nhead
        self.dh = dim // nhead
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, t):
        B, N, C = t.shape
        q, k, v = self.qkv(self.norm(t)).chunk(3, dim=-1)
        q = q.reshape(B, N, self.nhead, self.dh).transpose(1, 2)
        k = k.reshape(B, N, self.nhead, self.dh).transpose(1, 2)
        v = v.reshape(B, N, self.nhead, self.dh).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.dh ** 0.5), dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, N, C)
        return t + self.out(o)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.pos = nn.Conv2d(2, WIDTH, 1)
        self.attn = Attn(WIDTH, NHEAD)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.ctx = nn.Linear(WIDTH, WIDTH)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * WIDTH, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in self.fps])
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        f = self.body(obs)
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype)
        coords = torch.stack([ys.view(Y, 1).expand(Y, X), xs.view(1, X).expand(Y, X)], 0).unsqueeze(0)
        f = f + self.pos(coords)
        t = f.flatten(2).transpose(1, 2)
        t = self.attn(t)
        f = t.transpose(1, 2).reshape(B, -1, Y, X)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        h = F.relu(f + self.ctx(g)[:, :, None, None])
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            mx, mn = fp_pool(h, dy, dx)
            outs.append(head(torch.cat([h, mx, mn], 1)).flatten(1))
        logits = torch.cat(outs + [self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
