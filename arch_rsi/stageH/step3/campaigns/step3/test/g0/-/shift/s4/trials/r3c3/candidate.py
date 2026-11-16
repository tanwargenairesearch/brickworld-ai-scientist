import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "lr": 0.001, "ent_coef": 0.01, "gamma": 0.98, "gae_lambda": 0.95}

D = 48
NH = 4
HM = 32
G = 128


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = int(spec.Y) * int(spec.X)
        self.A = int(spec.n_place)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + 2, 64, 1), nn.ReLU(),
            nn.Conv2d(64, D, 1), nn.ReLU())
        self.convs = nn.ModuleList([nn.Conv2d(D, D, 3, padding=1) for _ in range(2)])
        self.pos = nn.Parameter(torch.randn(1, n, D) * 0.02)
        self.kv = nn.Linear(D, 2 * D)
        self.q = nn.Parameter(torch.randn(self.A, D) * 0.1)
        self.aemb = nn.Parameter(torch.randn(self.A, D) * 0.02)
        self.m1 = nn.Linear(D, HM)
        self.m2 = nn.Linear(HM, 1)
        nn.init.orthogonal_(self.m2.weight, 0.01)
        nn.init.zeros_(self.m2.bias)
        self.glob = nn.Sequential(nn.Linear(2 * D + int(spec.n_actions), G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        N = Y * X
        A = self.A
        dh = D // NH
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        h = self.stem(torch.cat([obs, ys, xs], dim=1))
        for c in self.convs:
            h = F.relu(h + c(h))
        t = h.flatten(2).transpose(1, 2) + self.pos
        kv = self.kv(t)
        k, v = kv.chunk(2, dim=-1)
        k = k.reshape(B, N, NH, dh).permute(0, 2, 1, 3)
        v = v.reshape(B, N, NH, dh).permute(0, 2, 1, 3)
        q = self.q.view(A, NH, dh).permute(1, 0, 2)
        sc = torch.matmul(q.unsqueeze(0), k.transpose(-1, -2)) / math.sqrt(dh)
        att = torch.softmax(sc, dim=-1)
        o = torch.matmul(att, v)
        o = o.permute(0, 2, 1, 3).reshape(B, A, D)
        z = F.relu(self.m1(F.relu(o + self.aemb)))
        place = self.m2(z).squeeze(-1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3)), mask.to(obs.dtype)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
