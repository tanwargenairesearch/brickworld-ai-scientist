import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

H1 = 256
HM = 128
H2 = 256
OUT_GAIN = 0.01


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        n_in = int(spec.in_channels) * int(spec.Y) * int(spec.X)
        self.obs_fc = nn.Sequential(nn.Linear(n_in, H1), nn.LayerNorm(H1), nn.ReLU())
        self.mask_fc = nn.Sequential(nn.Linear(self.n_place, HM), nn.LayerNorm(HM), nn.ReLU())
        self.body = nn.Sequential(nn.Linear(H1 + HM, H2), nn.LayerNorm(H2), nn.ReLU(),
                                  nn.Linear(H2, H2), nn.ReLU())
        self.pi = nn.Linear(H2, self.n_actions)
        self.v = nn.Linear(H2, 1)
        for m in (self.pi, self.v):
            nn.init.orthogonal_(m.weight, gain=OUT_GAIN)
            nn.init.zeros_(m.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        h = self.obs_fc(obs.reshape(B, -1))
        mh = self.mask_fc(mask[:, :self.n_place].to(obs.dtype))
        z = self.body(torch.cat([h, mh], dim=1))
        return self.pi(z), self.v(z).squeeze(1)


def build_policy(spec):
    return Net(spec)
