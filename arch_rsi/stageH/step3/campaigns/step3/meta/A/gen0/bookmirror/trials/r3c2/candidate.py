import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16}

STEM = 64
W = 32
H = 4
DK = 8
DV = 16
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = int(spec.in_channels)
        self.N = int(spec.Y) * int(spec.X)
        self.A = int(spec.n_place)
        self.stem = nn.Sequential(
            nn.Conv2d(C, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.kp = nn.Linear(2 * W, H * DK)
        self.vp = nn.Linear(2 * W, H * DV)
        self.pos = nn.Parameter(torch.randn(self.N, H * DK))
        self.q = nn.Parameter(torch.randn(self.A, H, DK))
        self.rp = nn.Linear(H * DV, HID)
        self.qp = nn.Linear(H * DK, HID)
        self.out = nn.Linear(HID, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(4 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        s = self.stem(obs)
        t = F.relu(self.c1(s))
        t = F.relu(t + self.c2(t))
        f = torch.cat([s, t], dim=1)
        tok = f.flatten(2).transpose(1, 2)
        k = (self.kp(tok) + self.pos).reshape(B, self.N, H, DK)
        v = self.vp(tok).reshape(B, self.N, H, DV)
        sc = torch.einsum('ahd,bnhd->bhan', self.q, k) * (1.0 / math.sqrt(DK))
        att = torch.softmax(sc, dim=-1)
        r = torch.einsum('bhan,bnhd->bahd', att, v).reshape(B, self.A, H * DV)
        qh = self.qp(self.q.reshape(self.A, H * DK))
        h = F.relu(self.rp(r) + qh)
        place = self.out(h).squeeze(-1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
