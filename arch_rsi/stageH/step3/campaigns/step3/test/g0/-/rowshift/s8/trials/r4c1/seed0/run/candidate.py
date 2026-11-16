import torch
import torch.nn as nn
import torch.nn.functional as F

D = 64
TH = 64
CH = 96
NBLK = 3
GH = 64

RECIPE = {"ppo_epochs": 1, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32}


class Block(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.t1 = nn.Linear(n, TH)
        self.t2 = nn.Linear(TH, n)
        self.ln2 = nn.LayerNorm(d)
        self.c1 = nn.Linear(d, CH)
        self.c2 = nn.Linear(CH, d)

    def forward(self, h):
        u = self.ln1(h).transpose(1, 2)
        u = self.t2(F.gelu(self.t1(u))).transpose(1, 2)
        h = h + u
        h = h + self.c2(F.gelu(self.c1(self.ln2(h))))
        return h


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        n = int(spec.Y) * int(spec.X)
        cin = int(spec.in_channels) + self.T
        self.embed = nn.Linear(cin, D)
        self.blocks = nn.ModuleList([Block(n, D) for _ in range(NBLK)])
        self.lnf = nn.LayerNorm(D)
        self.route = nn.Linear(n, n)
        self.head = nn.Sequential(nn.Linear(D, GH), nn.ReLU(), nn.Linear(GH, self.T))
        self.glob = nn.Sequential(nn.Linear(2 * D, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.T, Y, X)
        x = torch.cat([obs, m], dim=1).flatten(2).transpose(1, 2)
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        h = self.lnf(h)
        r = h + self.route(h.transpose(1, 2)).transpose(1, 2)
        place = self.head(r).transpose(1, 2).reshape(B, -1)
        g = self.glob(torch.cat([h.mean(1), h.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
