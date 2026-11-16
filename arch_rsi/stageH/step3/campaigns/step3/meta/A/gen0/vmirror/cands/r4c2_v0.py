import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "lr": 0.002, "gamma": 0.98, "gae_lambda": 0.95, "ent_coef": 0.003}

STEM = 64
COL = 48
WIDTH = 48
DRED = 16
MF = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.red = nn.Conv2d(WIDTH, DRED, 1)
        self.mfeat = nn.Linear(self.n_place, MF)
        self.dense = nn.Linear(DRED * self.Y * self.X + MF, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + MF, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        nn.init.normal_(self.dense.weight, std=0.01)
        nn.init.zeros_(self.dense.bias)

    def forward(self, obs, mask):
        m = mask[:, :self.n_place].to(obs.dtype)
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        mf = F.relu(self.mfeat(m))
        z = torch.cat([F.relu(self.red(f)).flatten(1), mf], dim=1)
        place = self.dense(z)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), mf], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
