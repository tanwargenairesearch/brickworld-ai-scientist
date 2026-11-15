import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "ent_coef": 0.003}

D = 48
NH = 4
NL = 2
HH = 32
GH = 64


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, h, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        y = self.n1(x)
        a, _ = self.att(y, y, y, need_weights=False)
        x = x + a
        x = x + self.ff(self.n2(x))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        Y = int(spec.Y)
        X = int(spec.X)
        cin = int(spec.in_channels) + self.T + 2
        self.embed = nn.Conv2d(cin, D, 3, padding=1)
        self.pos = nn.Parameter(torch.randn(1, Y * X, D) * 0.02)
        self.blocks = nn.ModuleList([Block(D, NH) for _ in range(NL)])
        self.nf = nn.LayerNorm(D)
        self.h1 = nn.Conv2d(D, HH, 3, padding=1)
        self.h2 = nn.Conv2d(HH, self.T, 1)
        self.glob = nn.Sequential(nn.Linear(2 * D, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        T = self.T
        m = mask[:, :T * Y * X].reshape(B, T, Y, X).to(obs.dtype)
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        e = self.embed(torch.cat([obs, m, yy, xx], dim=1))
        t = e.flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            t = blk(t)
        t = self.nf(t)
        f = t.transpose(1, 2).reshape(B, D, Y, X)
        place = self.h2(F.relu(self.h1(f))).flatten(1)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
