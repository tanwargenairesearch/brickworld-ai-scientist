import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.97, "gae_lambda": 0.95, "lr": 0.0007, "ent_coef": 0.003, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32}

D = 64
NH = 4
FF = 128
NL = 2
HID = 32
GH = 64


class Layer(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = nn.Linear(ff, d)

    def forward(self, t):
        B, N, Dm = t.shape
        q, k, v = self.qkv(self.ln1(t)).chunk(3, dim=-1)
        hd = Dm // self.h
        q = q.reshape(B, N, self.h, hd).transpose(1, 2)
        k = k.reshape(B, N, self.h, hd).transpose(1, 2)
        v = v.reshape(B, N, self.h, hd).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd), dim=-1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(B, N, Dm)
        t = t + self.o(y)
        t = t + self.ff2(F.relu(self.ff1(self.ln2(t))))
        return t


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.embed = nn.Linear(spec.in_channels, D)
        self.pos = nn.Parameter(torch.randn(self.Y * self.X, D) * 0.02)
        self.layers = nn.ModuleList([Layer(D, NH, FF) for _ in range(NL)])
        self.lnf = nn.LayerNorm(D)
        self.fconv = nn.ModuleList([nn.Conv2d(D, HID, (dy, dx)) for dy, dx in self.fps])
        self.fout = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * D, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        t = obs.flatten(2).transpose(1, 2)
        t = self.embed(t) + self.pos.unsqueeze(0)
        for layer in self.layers:
            t = layer(t)
        t = self.lnf(t)
        h = t.transpose(1, 2).reshape(B, D, self.Y, self.X)
        res = []
        for (dy, dx), fc, fo in zip(self.fps, self.fconv, self.fout):
            q = F.pad(h, (0, dx - 1, 0, dy - 1))
            res.append(fo(F.relu(fc(q))))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
