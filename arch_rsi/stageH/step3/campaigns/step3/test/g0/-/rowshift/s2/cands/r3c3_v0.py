import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 64, "rollout_len": 16, "n_minibatches": 16}

W = 48
P = 8
HH = 32
K = 4


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.nty = int(spec.n_types)
        self.npl = int(spec.n_place)
        self.sq = self.Y == self.X
        self.nv = 8 if self.sq else 4
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + self.nty, 64, 1), nn.ReLU(),
            nn.Conv2d(64, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.ba = nn.Conv2d(W, W, 3, padding=2, dilation=2)
        self.bb = nn.Conv2d(W, W, 3, padding=1)
        self.proj = nn.Conv2d(W, P, 1)
        self.h1 = nn.Conv2d(P * self.nv, HH, 3, padding=1)
        self.h2 = nn.Conv2d(HH, self.nty, 1)
        self.h3 = nn.Conv2d(HH, K, 1)
        self.remap = nn.Linear(K * self.Y * self.X, self.npl)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def views(self, p):
        v = [p, torch.flip(p, [3]), torch.flip(p, [2]), torch.flip(p, [2, 3])]
        if self.sq:
            t = p.transpose(2, 3)
            v = v + [t, torch.flip(t, [3]), torch.flip(t, [2]), torch.flip(t, [2, 3])]
        return torch.cat(v, dim=1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.npl].to(obs.dtype).reshape(B, self.nty, self.Y, self.X)
        s = self.stem(torch.cat([obs, m], dim=1))
        x = F.relu(self.c1(s))
        x = F.relu(x + self.bb(F.relu(self.ba(x))))
        p = F.relu(self.proj(x))
        h = F.relu(self.h1(self.views(p)))
        aligned = self.h2(h).flatten(1)
        r = self.remap(F.relu(self.h3(h)).flatten(1))
        place = aligned + r
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
