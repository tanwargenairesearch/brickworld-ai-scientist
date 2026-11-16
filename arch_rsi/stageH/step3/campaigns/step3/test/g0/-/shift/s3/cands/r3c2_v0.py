import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 32, "n_minibatches": 8, "ppo_epochs": 2, "lr": 0.001, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005}

C1 = 64
C2 = 32
C3 = 16
HD = 384


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = len(spec.footprints)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        cin = int(spec.in_channels) + self.T
        self.enc = nn.Sequential(
            nn.Conv2d(cin, C1, 1), nn.ReLU(),
            nn.Conv2d(C1, C2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(C2, C3, 3, padding=1), nn.ReLU())
        self.mlp = nn.Sequential(
            nn.Linear(C3 * self.Y * self.X, HD), nn.ReLU(),
            nn.Linear(HD, HD), nn.ReLU())
        self.pi = nn.Linear(HD, int(spec.n_actions))
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        self.value_head = nn.Linear(HD, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.n_place].reshape(B, self.T, self.Y, self.X).to(obs.dtype)
        f = self.enc(torch.cat([obs, m], 1))
        h = self.mlp(f.flatten(1))
        return self.pi(h), self.value_head(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
