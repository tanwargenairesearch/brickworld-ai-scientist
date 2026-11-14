import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.97, "gae_lambda": 0.95}

W = 48
R = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.stem = nn.Conv2d(c, W, 1)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.red = nn.Conv2d(W, R, 1)
        self.act = nn.Linear(R * int(spec.Y) * int(spec.X), int(spec.n_actions))
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        a = F.relu(self.stem(obs))
        x = F.relu(a + self.c1(a))
        f = F.relu(x + self.c2(x))
        r = F.relu(self.red(f))
        logits = self.act(r.flatten(1))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
