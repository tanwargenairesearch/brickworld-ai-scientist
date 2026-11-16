import torch
import torch.nn as nn
import torch.nn.functional as F

H = 320

RECIPE = {"ppo_epochs": 1, "gamma": 0.97, "gae_lambda": 0.95, "lr": 0.001, "ent_coef": 0.003, "n_minibatches": 16}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y, X = int(spec.Y), int(spec.X)
        n_in = int(spec.in_channels) * Y * X
        self.obs_in = nn.Linear(n_in, H)
        self.mask_in = nn.Linear(int(spec.n_actions), H)
        self.l2 = nn.Linear(H, H)
        self.pi = nn.Linear(H, int(spec.n_actions))
        self.v1 = nn.Linear(H, 64)
        self.v2 = nn.Linear(64, 1)
        for m in [self.obs_in, self.mask_in, self.l2, self.v1]:
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v2.weight, gain=1.0)
        nn.init.zeros_(self.v2.bias)

    def forward(self, obs, mask):
        h = F.relu(self.obs_in(obs.flatten(1)) + self.mask_in(mask.to(obs.dtype)))
        h = F.relu(self.l2(h))
        logits = self.pi(h)
        value = self.v2(F.relu(self.v1(h))).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)


import math
