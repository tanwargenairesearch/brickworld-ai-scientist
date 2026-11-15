import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 64, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
HEADS = 4
PROJ = 32
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.ln1 = nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.ao = nn.Linear(WIDTH, WIDTH)
        self.ln2 = nn.LayerNorm(WIDTH)
        self.ff = nn.Sequential(nn.Linear(WIDTH, 2 * WIDTH), nn.ReLU(), nn.Linear(2 * WIDTH, WIDTH))
        self.proj = nn.Conv2d(WIDTH + COL, PROJ, 1)
        self.fconv = nn.ModuleList([nn.Conv2d(PROJ, HID, (dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        t = f.flatten(2).transpose(1, 2)
        N = t.shape[1]
        dh = WIDTH // HEADS
        h = self.ln1(t)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.reshape(B, N, HEADS, dh).transpose(1, 2)
        k = k.reshape(B, N, HEADS, dh).transpose(1, 2)
        v = v.reshape(B, N, HEADS, dh).transpose(1, 2)
        att = torch.softmax(torch.matmul(q, k.transpose(2, 3)) / math.sqrt(dh), dim=-1)
        o = torch.matmul(att, v).transpose(1, 2).reshape(B, N, WIDTH)
        t = t + self.ao(o)
        t = t + self.ff(self.ln2(t))
        g2 = t.transpose(1, 2).reshape(B, WIDTH, Y, X)
        p = F.relu(self.proj(torch.cat([g2, s], dim=1)))
        res = []
        for (dy, dx), conv, out in zip(self.fps, self.fconv, self.outs):
            qq = F.pad(p, (0, dx - 1, 0, dy - 1))
            hh = F.relu(conv(qq))
            res.append(out(hh))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([g2.mean((2, 3)), g2.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
