import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "lr": 0.001, "ent_coef": 0.005}

H1 = 320
H2 = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        d = int(spec.in_channels) * int(spec.Y) * int(spec.X)
        self.lo = nn.Linear(d, H1)
        self.lm = nn.Linear(int(spec.n_actions), H1, bias=False)
        self.n1 = nn.LayerNorm(H1)
        self.l2 = nn.Linear(H1, H2)
        self.n2 = nn.LayerNorm(H2)
        self.pi = nn.Linear(H2, int(spec.n_actions))
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        self.v = nn.Linear(H2, 1)

    def forward(self, obs, mask):
        h = self.lo(obs.flatten(1)) + self.lm(mask.to(obs.dtype))
        h = F.relu(self.n1(h))
        h = F.relu(self.n2(self.l2(h)))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
