import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.002, "ent_coef": 0.001, "clip_eps": 0.3, "gamma": 0.97, "gae_lambda": 0.95}

D = 40
HEADS = 4
PROJ = 32
HID = 32
GH = 64


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        self.embed = nn.Conv2d(spec.in_channels, D, 1)
        self.local = nn.Conv2d(D, D, 3, padding=1)
        self.pos = nn.Parameter(torch.randn(1, Y * X, D) * 0.02)
        self.ln1 = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, 2 * D), nn.ReLU(), nn.Linear(2 * D, D))
        self.head = PoolHead(D, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * D, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        x = F.relu(self.embed(obs))
        x = F.relu(x + self.local(x))
        t = x.flatten(2).transpose(1, 2) + self.pos
        h = self.ln1(t)
        a, _ = self.attn(h, h, h, need_weights=False)
        t = t + a
        t = t + self.mlp(self.ln2(t))
        f = t.transpose(1, 2).reshape(B, D, Y, X)
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
