import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
P = 12
HID = 24
HEADS = 4


class AttnBlock(nn.Module):
    def __init__(self, w, heads):
        super().__init__()
        self.h = heads
        self.d = w // heads
        self.pos = nn.Linear(2, w)
        self.ln1 = nn.LayerNorm(w)
        self.qkv = nn.Linear(w, 3 * w)
        self.o = nn.Linear(w, w)
        self.ln2 = nn.LayerNorm(w)
        self.ff1 = nn.Linear(w, 2 * w)
        self.ff2 = nn.Linear(2 * w, w)

    def forward(self, f):
        B, W, Y, X = f.shape
        N = Y * X
        ys = torch.linspace(-1.0, 1.0, Y, device=f.device, dtype=f.dtype)
        xs = torch.linspace(-1.0, 1.0, X, device=f.device, dtype=f.dtype)
        gy = ys.reshape(Y, 1).expand(Y, X)
        gx = xs.reshape(1, X).expand(Y, X)
        coords = torch.stack([gy, gx], dim=-1).reshape(1, N, 2)
        t = f.flatten(2).transpose(1, 2) + self.pos(coords)
        z = self.ln1(t)
        qkv = self.qkv(z).reshape(B, N, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d), dim=-1)
        a = torch.matmul(att, v).transpose(1, 2).reshape(B, N, W)
        t = t + self.o(a)
        t = t + self.ff2(F.relu(self.ff1(self.ln2(t))))
        return t.transpose(1, 2).reshape(B, W, Y, X)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.attn = AttnBlock(WIDTH, HEADS)
        self.layer_in = nn.Conv2d(7 + self.Z, P, 1)
        self.layer_ctx = nn.Conv2d(WIDTH, self.Z * P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.base = nn.Conv2d(WIDTH, len(self.fps), 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        x = F.relu(x + self.c2(x))
        f = self.attn(x)

        per = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 2, 1, 3, 4)
        shared = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype).reshape(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        lin = torch.cat([per, shared, eye], dim=2).reshape(B * Z, 7 + Z, Y, X)
        a = self.layer_in(lin)
        c = self.layer_ctx(f).reshape(B, Z, P, Y, X).reshape(B * Z, P, Y, X)
        g = F.relu(a + c)

        base = self.base(f)
        res = []
        for t, ((dy, dx), mix, out) in enumerate(zip(self.fps, self.mix, self.outs)):
            if dy * dx == 1:
                mx, mn, av = g, g, g
            else:
                q = F.pad(g, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            sc = out(h).reshape(B, Z, Y, X)
            sc = torch.logsumexp(sc, dim=1) + base[:, t]
            res.append(sc)
        place = torch.stack(res, dim=1).flatten(1)

        gl = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(gl)], dim=1)
        return logits, self.value_head(gl).squeeze(1)


def build_policy(spec):
    return Net(spec)
