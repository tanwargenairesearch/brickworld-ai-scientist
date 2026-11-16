import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 32, "n_minibatches": 16, "lr": 0.0007, "ent_coef": 0.003, "gamma": 0.99, "gae_lambda": 0.95}

D = 64
HEADS = 4
HID = 32
GLOB = 64


class AttnLayer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.nh = heads
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ff1 = nn.Linear(d, 2 * d)
        self.ff2 = nn.Linear(2 * d, d)

    def forward(self, x):
        B, N, Dm = x.shape
        hd = Dm // self.nh
        y = self.ln1(x)
        qkv = self.qkv(y).view(B, N, 3, self.nh, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd), dim=-1)
        z = torch.matmul(a, v).transpose(1, 2).reshape(B, N, Dm)
        x = x + self.o(z)
        x = x + self.ff2(F.relu(self.ff1(self.ln2(x))))
        return x


class FootHead(nn.Module):
    def __init__(self, c, footprints, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.convs = nn.ModuleList([nn.Conv2d(c, hid, (dy, dx)) for dy, dx in self.fps])
        self.mids = nn.ModuleList([nn.Conv2d(hid, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.orthogonal_(o.weight, gain=0.01)
            nn.init.zeros_(o.bias)

    def forward(self, f):
        res = []
        for (dy, dx), cv, md, out in zip(self.fps, self.convs, self.mids, self.outs):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            h = F.relu(cv(q))
            h = F.relu(md(h))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.inp = nn.Conv2d(spec.in_channels + 2, D, 1)
        self.conv = nn.Conv2d(D, D, 3, padding=1)
        self.attn = AttnLayer(D, HEADS)
        self.head = FootHead(D, spec.footprints, HID)
        self.glob = nn.Sequential(nn.Linear(2 * D, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        nn.init.orthogonal_(self.done_head.weight, gain=0.01)
        nn.init.zeros_(self.done_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        h = F.relu(self.inp(torch.cat([obs, ys, xs], dim=1)))
        h = F.relu(h + self.conv(h))
        t = h.flatten(2).transpose(1, 2)
        t = self.attn(t)
        f = t.transpose(1, 2).reshape(B, D, Y, X)
        place = self.head(f)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
