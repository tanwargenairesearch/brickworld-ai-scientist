import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
W = 48
PR = 16
H = 512
GH = 64

RECIPE = {"ppo_epochs": 2, "gamma": 0.98, "gae_lambda": 0.95, "ent_coef": 0.003, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_cells = int(spec.Y) * int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.proj = nn.Conv2d(W, PR, 1)
        self.fc1 = nn.Linear(PR * n_cells, H)
        self.fc2 = nn.Linear(H, int(spec.n_place))
        nn.init.orthogonal_(self.fc2.weight, 0.01)
        nn.init.zeros_(self.fc2.bias)
        self.done_head = nn.Linear(H, 1)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU(), nn.Linear(GH, 1))

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        z = F.relu(self.fc1(F.relu(self.proj(f)).flatten(1)))
        logits = torch.cat([self.fc2(z), self.done_head(z)], dim=1)
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
