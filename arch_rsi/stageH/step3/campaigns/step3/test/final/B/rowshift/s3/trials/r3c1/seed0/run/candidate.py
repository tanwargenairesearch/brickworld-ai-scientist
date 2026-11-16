import math
import torch
import torch.nn as nn
import torch.nn.functional as F

HID = 256
USE_LN = 0
RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}


def ortho(m, g):
    nn.init.orthogonal_(m.weight, g)
    nn.init.zeros_(m.bias)
    return m


class Dense(nn.Module):
    def __init__(self, i, o, use_ln):
        super().__init__()
        self.fc = ortho(nn.Linear(i, o), math.sqrt(2.0))
        self.use_ln = use_ln
        if use_ln:
            self.ln = nn.LayerNorm(o)

    def forward(self, x):
        h = self.fc(x)
        if self.use_ln:
            h = self.ln(h)
        return F.relu(h)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_obs = int(spec.in_channels) * int(spec.Y) * int(spec.X)
        n_act = int(spec.n_actions)
        self.obs_fc = ortho(nn.Linear(n_obs, HID), math.sqrt(2.0))
        self.mask_fc = ortho(nn.Linear(n_act, HID), math.sqrt(2.0))
        self.use_ln = USE_LN
        if USE_LN:
            self.ln0 = nn.LayerNorm(HID)
        self.body = Dense(HID, HID, USE_LN)
        self.pi_h = Dense(HID, HID, USE_LN)
        self.v_h = Dense(HID, HID, USE_LN)
        self.pi = ortho(nn.Linear(HID, n_act), 0.01)
        self.v = ortho(nn.Linear(HID, 1), 1.0)

    def forward(self, obs, mask):
        h = self.obs_fc(obs.flatten(1)) + self.mask_fc(mask.to(obs.dtype))
        if self.use_ln:
            h = self.ln0(h)
        h = self.body(F.relu(h))
        logits = self.pi(self.pi_h(h))
        value = self.v(self.v_h(h)).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
