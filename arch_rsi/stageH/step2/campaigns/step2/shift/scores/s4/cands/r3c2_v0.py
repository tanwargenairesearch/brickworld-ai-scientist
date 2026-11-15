import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "gamma": 0.95, "gae_lambda": 0.95}

STEM = 48
WIDTH = 32
RED = 24
GH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = int(spec.in_channels)
        N = int(spec.Y) * int(spec.X)
        P = int(spec.n_place)
        hid = max(64, min(384, int(700000 / (RED * N + P + GH))))
        self.stem = nn.Sequential(nn.Conv2d(C, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, STEM, 1), nn.ReLU())
        self.c1 = nn.Conv2d(STEM, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.red = nn.Conv2d(WIDTH, RED, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.fc1 = nn.Linear(RED * N + GH, hid)
        self.fc2 = nn.Linear(hid, P)
        nn.init.orthogonal_(self.fc2.weight, gain=0.01)
        nn.init.zeros_(self.fc2.bias)
        self.done_head = nn.Linear(hid, 1)
        self.value_head = nn.Linear(hid, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        r = F.relu(self.red(f)).flatten(1)
        z = F.relu(self.fc1(torch.cat([r, g], dim=1)))
        place = self.fc2(z)
        logits = torch.cat([place, self.done_head(z)], dim=1)
        return logits, self.value_head(z).squeeze(1)


def build_policy(spec):
    return Net(spec)
