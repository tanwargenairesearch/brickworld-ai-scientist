import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 64, "rollout_len": 32, "n_minibatches": 16, "ppo_epochs": 1, "lr": 0.001, "ent_coef": 0.003, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
W = 48
H = 16
G = 96


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.hid = nn.Conv2d(W, self.nt * H, 3, padding=1)
        self.out = nn.Conv2d(self.nt * H, self.nt, 1, groups=self.nt)
        self.glob = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        x = self.stem(obs)
        x = F.relu(x + self.c1(x))
        x = F.relu(x + self.c2(x))
        place = self.out(F.relu(self.hid(x))).flatten(1)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
