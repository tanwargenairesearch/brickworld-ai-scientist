import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0005, "ppo_epochs": 2, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ent_coef": 0.01, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
D = 32
NH = 4
HID = 64
NBLOCKS = 2


class Block(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.nh = nh
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))

    def forward(self, t):
        B, N, d = t.shape
        dh = d // self.nh
        h = self.qkv(self.ln1(t)).reshape(B, N, 3, self.nh, dh).permute(2, 0, 3, 1, 4)
        q, k, v = h[0], h[1], h[2]
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh), dim=-1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(B, N, d)
        t = t + self.o(y)
        return t + self.ff(self.ln2(t))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.P = int(spec.n_place)
        N = int(spec.Y) * int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, D, 3, padding=1))
        self.pos = nn.Parameter(torch.randn(1, N, D) * 0.5)
        self.blocks = nn.ModuleList([Block(D, NH) for _ in range(NBLOCKS)])
        self.lnf = nn.LayerNorm(D)
        self.aq = nn.Parameter(torch.randn(self.P, D) * 0.5)
        self.kv = nn.Linear(D, 2 * D)
        self.ao = nn.Linear(D, HID)
        self.ae = nn.Linear(D, HID)
        self.aout = nn.Linear(HID, 1)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.act_head = nn.Linear(64, int(spec.n_actions))
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        f = self.stem(obs)
        t = f.flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            t = blk(t)
        t = self.lnf(t)
        N = t.shape[1]
        dh = D // NH
        kv = self.kv(t).reshape(B, N, 2, NH, dh).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.aq.reshape(self.P, NH, dh).transpose(0, 1).unsqueeze(0)
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh), dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, self.P, D)
        h = F.relu(self.ao(o) + self.ae(self.aq))
        place = self.aout(h).squeeze(-1)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1) + self.act_head(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
