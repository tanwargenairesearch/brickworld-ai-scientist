import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 1e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "ent_coef": 0.01, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
RED = 16
MF = 64
G = 128
H = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = int(spec.Y) * int(spec.X)
        na = int(spec.n_actions)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.red = nn.Conv2d(WIDTH, RED, 1)
        self.mask_in = nn.Linear(na, MF)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + MF, G), nn.ReLU())
        self.hid = nn.Sequential(nn.Linear(RED * n + G, H), nn.ReLU())
        self.pi = nn.Linear(H, na)
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        mfeat = F.relu(self.mask_in(mask.to(obs.dtype)))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), mfeat], dim=1))
        r = F.relu(self.red(f)).flatten(1)
        h = self.hid(torch.cat([r, g], dim=1))
        logits = self.pi(h)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
