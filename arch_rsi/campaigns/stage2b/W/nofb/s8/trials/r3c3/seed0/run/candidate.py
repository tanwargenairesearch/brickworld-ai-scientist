import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 3, "gae_lambda": 0.95}

W = 64
P = 32
H = 32
NH = 4


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + 2, W, 1), nn.ReLU(),
            nn.Conv2d(W, W, 1), nn.ReLU())
        self.ctx = nn.Conv2d(W, W, 3, padding=1)
        self.ln = nn.LayerNorm(W)
        self.qkv = nn.Linear(W, 3 * W)
        self.att_out = nn.Linear(W, W)
        self.proj = nn.Conv2d(W, P, 1)
        heads = []
        for _ in self.footprints:
            out = nn.Conv2d(H, 1, 1)
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
            heads.append(nn.Sequential(nn.Conv2d(3 * P, H, 1), nn.ReLU(), out))
        self.heads = nn.ModuleList(heads)
        self.glob = nn.Sequential(nn.Linear(2 * W, W), nn.ReLU())
        self.done_head = nn.Linear(W, 1)
        self.value_head = nn.Linear(W, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        f = self.stem(torch.cat([obs, ys, xs], dim=1))
        f = f + F.relu(self.ctx(f))
        N = Y * X
        D = W // NH
        t = f.flatten(2).transpose(1, 2)
        h = self.ln(t)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.reshape(B, N, NH, D).transpose(1, 2)
        k = k.reshape(B, N, NH, D).transpose(1, 2)
        v = v.reshape(B, N, NH, D).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D), dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, N, W)
        t = t + self.att_out(o)
        f = t.transpose(1, 2).reshape(B, W, Y, X)
        g = F.relu(self.proj(f))
        outs = []
        for fp, head in zip(self.footprints, self.heads):
            dy, dx = fp
            gp = F.pad(g, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(gp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-gp, (dy, dx), stride=1)
            av = F.avg_pool2d(gp, (dy, dx), stride=1)
            outs.append(head(torch.cat([mx, mn, av], dim=1)).flatten(1))
        place = torch.cat(outs, dim=1)
        gl = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(gl)], dim=1)
        return logits, self.value_head(gl).squeeze(1)


def build_policy(spec):
    return Net(spec)
