import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "lr": 0.001, "ent_coef": 0.01, "gamma": 0.95, "gae_lambda": 0.9}

STEM = 64
COL = 48
WIDTH = 40
K = 16
HD = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        YX = int(spec.Y) * int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.red = nn.Conv2d(COL + WIDTH, K, 1)
        self.fc1 = nn.Linear(K * YX, HD)
        self.fc2 = nn.Linear(HD, int(spec.n_place))
        self.glob = nn.Sequential(nn.Linear(2 * (COL + WIDTH) + HD, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        z = torch.cat([s, f], dim=1)
        r = F.relu(self.red(z)).flatten(1)
        h = F.relu(self.fc1(r))
        place = self.fc2(h)
        g = self.glob(torch.cat([z.mean((2, 3)), z.amax((2, 3)), h], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
