import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 64, "rollout_len": 32, "n_minibatches": 16, "lr": 0.0007, "ent_coef": 0.002, "gamma": 0.99, "gae_lambda": 0.95}

W = 48
HH = 32
VW = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.nty = int(spec.n_types)
        self.npl = int(spec.n_place)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + self.nty, 64, 1), nn.ReLU(),
            nn.Conv2d(64, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.ba = nn.Conv2d(W, W, 3, padding=2, dilation=2)
        self.bb = nn.Conv2d(W, W, 3, padding=1)
        self.h1 = nn.Conv2d(W, HH, 3, padding=1)
        self.h2 = nn.Conv2d(HH, self.nty, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.v1 = nn.Conv2d(spec.in_channels, VW, 1)
        self.v2 = nn.Conv2d(VW, VW, 3, padding=1)
        self.v3 = nn.Conv2d(VW, VW, 3, padding=2, dilation=2)
        self.vmlp = nn.Sequential(nn.Linear(2 * VW, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.npl].to(obs.dtype).reshape(B, self.nty, self.Y, self.X)
        s = self.stem(torch.cat([obs, m], dim=1))
        x = F.relu(self.c1(s))
        x = F.relu(x + self.bb(F.relu(self.ba(x))))
        h = F.relu(self.h1(x))
        place = self.h2(h).flatten(1)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        v = F.relu(self.v1(obs))
        v = F.relu(self.v2(v))
        v = F.relu(v + self.v3(v))
        value = self.vmlp(torch.cat([v.mean((2, 3)), v.amax((2, 3))], dim=1)).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
