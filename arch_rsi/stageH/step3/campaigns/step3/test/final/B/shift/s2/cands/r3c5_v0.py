import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

HID = 288


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_in = int(spec.in_channels) * int(spec.Y) * int(spec.X)
        na = int(spec.n_actions)
        self.fo = nn.Linear(n_in, HID)
        self.fm = nn.Linear(na, HID)
        self.ln0 = nn.LayerNorm(HID)
        self.l1 = nn.Linear(HID, HID)
        self.n1 = nn.LayerNorm(HID)
        self.l2 = nn.Linear(HID, HID)
        self.n2 = nn.LayerNorm(HID)
        self.pi = nn.Linear(HID, na)
        self.v1 = nn.Linear(HID, 64)
        self.v = nn.Linear(64, 1)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, obs, mask):
        h = F.relu(self.ln0(self.fo(obs.flatten(1)) + self.fm(mask.to(obs.dtype))))
        h = F.relu(self.n1(h + self.l1(h)))
        h = F.relu(self.n2(h + self.l2(h)))
        logits = self.pi(h)
        value = self.v(F.relu(self.v1(h))).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
