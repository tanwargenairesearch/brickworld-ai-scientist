import torch
import torch.nn as nn
import torch.nn.functional as F

CELL = 16
HID = 384

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_actions = int(spec.n_actions)
        cells = self.Y * self.X
        self.cell = nn.Conv2d(spec.in_channels, CELL, 1)
        self.cell_ln = nn.LayerNorm(CELL)
        self.obs_fc = nn.Linear(CELL * cells, HID)
        self.mask_fc = nn.Linear(self.n_actions, HID)
        self.ln1 = nn.LayerNorm(HID)
        self.fc2 = nn.Linear(HID, HID)
        self.ln2 = nn.LayerNorm(HID)
        self.pi = nn.Linear(HID, self.n_actions)
        self.v = nn.Linear(HID, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, 1.0)
        nn.init.zeros_(self.v.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        c = self.cell(obs).permute(0, 2, 3, 1)
        c = F.relu(self.cell_ln(c)).reshape(B, -1)
        h = self.obs_fc(c) + self.mask_fc(mask.float())
        h = F.relu(self.ln1(h))
        h = F.relu(self.ln2(h + self.fc2(h)))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
