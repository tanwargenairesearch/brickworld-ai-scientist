import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "ppo_epochs": 1, "ent_coef": 0.003}

W = 48
HH = 32
GH = 64
RED = 8


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        Y = int(spec.Y)
        X = int(spec.X)
        self.n_place = int(spec.n_place)
        cin = int(spec.in_channels) + self.T + 2
        self.stem = nn.Sequential(nn.Conv2d(cin, 64, 1), nn.ReLU(), nn.Conv2d(64, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.g1 = nn.Linear(2 * W, W)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.h1 = nn.Conv2d(W, HH, 3, padding=1)
        self.h2 = nn.Conv2d(HH, self.T, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU())
        self.red = nn.Conv2d(W, RED, 1)
        self.free = nn.Linear(RED * Y * X + GH, self.n_place)
        nn.init.zeros_(self.free.weight)
        nn.init.zeros_(self.free.bias)
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        T = self.T
        m = mask[:, :T * Y * X].reshape(B, T, Y, X).to(obs.dtype)
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        x = self.stem(torch.cat([obs, m, yy, xx], dim=1))
        x = F.relu(x + self.c1(x))
        g = F.relu(self.g1(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)))
        x = x + g[:, :, None, None]
        x = F.relu(x + self.c2(x))
        conv_place = self.h2(F.relu(self.h1(x))).flatten(1)
        g2 = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        r = F.relu(self.red(x)).flatten(1)
        place = conv_place + self.free(torch.cat([r, g2], dim=1))
        logits = torch.cat([place, self.done_head(g2)], dim=1)
        return logits, self.value_head(g2).squeeze(1)


def build_policy(spec):
    return Net(spec)
