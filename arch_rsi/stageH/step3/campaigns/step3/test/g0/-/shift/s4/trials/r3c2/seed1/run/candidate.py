import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "lr": 0.0005, "ent_coef": 0.01, "gamma": 0.98, "gae_lambda": 0.95}

H = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        nin = int(spec.in_channels) * int(spec.Y) * int(spec.X)
        na = int(spec.n_actions)
        self.fo = nn.Linear(nin, H)
        self.fm = nn.Linear(na, H)
        self.f2 = nn.Linear(H, H)
        self.pi = nn.Linear(H, na)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        self.v = nn.Linear(H, 1)

    def forward(self, obs, mask):
        x = F.relu(self.fo(obs.flatten(1)) + self.fm(mask.to(obs.dtype)))
        x = F.relu(x + self.f2(x))
        logits = self.pi(x)
        return logits, self.v(x).squeeze(1)


def build_policy(spec):
    return Net(spec)
