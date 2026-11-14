import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0015, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.98, "gae_lambda": 0.95}

STEM = 64
WIDTH = 40
HEAD = 48
RED = 16
FCH = 384


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = int(spec.Y) * int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.b1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.b2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.ctx = nn.Linear(2 * WIDTH, WIDTH)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h2 = nn.Conv2d(HEAD, spec.n_types, 1)
        nn.init.normal_(self.h2.weight, std=0.01)
        nn.init.zeros_(self.h2.bias)
        self.red = nn.Conv2d(WIDTH, RED, 1)
        self.fc1 = nn.Linear(RED * n, FCH)
        self.fc2 = nn.Linear(FCH, spec.n_place)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + FCH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = self.stem(obs)
        f = F.relu(f + self.b1(f))
        c = self.ctx(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        f = F.relu(f + c[:, :, None, None])
        f = F.relu(f + self.b2(f))
        conv_place = self.h2(F.relu(self.h1(f))).flatten(1)
        z = F.relu(self.fc1(F.relu(self.red(f)).flatten(1)))
        place = conv_place + self.fc2(z)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), z], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
