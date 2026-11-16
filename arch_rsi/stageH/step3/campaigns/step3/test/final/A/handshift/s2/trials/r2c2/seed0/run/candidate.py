import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 32
SKIP = 16
GH = 64
HEADS = 4
DH = 8
EMB = 32
QSCALE = 3.0


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = nn.Conv2d(cin, SKIP, 1)

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        f = torch.cat([x, F.relu(self.skip(obs))], dim=1)
        return x, f


class CellAttnHead(nn.Module):
    def __init__(self, c, spec):
        super().__init__()
        self.N = int(spec.Y) * int(spec.X)
        self.A = int(spec.n_actions)
        self.proj = nn.Conv2d(c, HEADS * DH, 1)
        self.q = nn.Parameter(0.1 * torch.randn(HEADS, self.A, self.N))
        self.inp = nn.Linear(HEADS * DH, EMB)
        self.emb = nn.Parameter(0.5 * torch.randn(self.A, EMB))
        self.gl = nn.Linear(GH, EMB)
        self.out = nn.Linear(EMB, 1)

    def forward(self, f, g):
        B = f.shape[0]
        v = self.proj(f).reshape(B, HEADS, DH, self.N)
        att = torch.softmax(QSCALE * self.q, dim=-1)
        r = torch.einsum('han,bhdn->bahd', att, v).reshape(B, self.A, HEADS * DH)
        h = F.relu(self.inp(r) + self.emb + self.gl(g).unsqueeze(1))
        return self.out(h).squeeze(-1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec.in_channels)
        self.head = CellAttnHead(WIDTH + SKIP, spec)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.value_head = nn.Linear(GH, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.head.out.weight, 0.01)
        nn.init.orthogonal_(self.value_head.weight, 1.0)

    def forward(self, obs, mask):
        x, f = self.trunk(obs)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = self.head(f, g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
