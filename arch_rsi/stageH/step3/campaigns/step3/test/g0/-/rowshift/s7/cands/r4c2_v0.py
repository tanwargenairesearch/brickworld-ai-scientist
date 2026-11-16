import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "lr": 0.0007, "ent_coef": 0.003, "gae_lambda": 0.95, "gamma": 0.97}

W = 48
HM = 32
DET = 16
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.n_types = len(spec.footprints)
        self.stem = nn.Conv2d(spec.in_channels + 2, W, 1)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=2, dilation=2)
        self.c3 = nn.Conv2d(W, W, 3, padding=1)
        self.hmix = nn.Conv2d(W, HM, 1)
        self.hout = nn.Conv2d(HM, self.n_types, 3, padding=1)
        self.det = nn.Conv2d(spec.in_channels, DET, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W + DET, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        h = F.relu(self.stem(torch.cat([obs, ys, xs], dim=1)))
        h = F.relu(h + self.c1(h))
        h = F.relu(h + self.c2(h))
        h = F.relu(h + self.c3(h))
        place = self.hout(F.relu(self.hmix(h))).flatten(1)
        d = self.det(obs).amax((2, 3))
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3)), d], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
