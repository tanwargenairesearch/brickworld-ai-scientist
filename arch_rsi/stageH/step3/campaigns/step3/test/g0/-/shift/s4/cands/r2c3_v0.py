import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "lr": 0.0007, "ent_coef": 0.01, "gamma": 0.98, "gae_lambda": 0.95}

D = 64
NH = 4
NL = 2


class Block(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.nh = nh
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        B, N, Dm = x.shape
        dh = Dm // self.nh
        qkv = self.qkv(self.ln1(x)).view(B, N, 3, self.nh, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh), dim=-1)
        y = torch.matmul(att, v).transpose(1, 2).reshape(B, N, Dm)
        x = x + self.o(y)
        return x + self.ff(self.ln2(x))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        self.embed = nn.Conv2d(spec.in_channels + 2, D, 3, padding=1)
        self.blocks = nn.ModuleList([Block(D, NH) for _ in range(NL)])
        self.lnf = nn.LayerNorm(D)
        self.hmix = nn.Conv2d(D, D, 1)
        self.out = nn.Conv2d(D, self.nt, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        e = self.embed(torch.cat([obs, ys, xs], dim=1))
        t = e.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            t = blk(t)
        t = self.lnf(t)
        grid = t.transpose(1, 2).reshape(B, D, Y, X)
        z = F.relu(self.hmix(grid))
        place = self.out(z).flatten(1)
        pooled = torch.cat([t.mean(1), t.amax(1)], dim=1)
        g = self.glob(pooled)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
