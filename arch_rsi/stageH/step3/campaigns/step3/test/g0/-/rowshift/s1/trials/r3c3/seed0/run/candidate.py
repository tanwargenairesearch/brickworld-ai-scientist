import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 64, "rollout_len": 32, "n_minibatches": 8, "lr": 0.0005, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01}

W = 40
H = 32


def add_coords(obs):
    B, C, Y, X = obs.shape
    ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
    xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
    return torch.cat([obs, ys, xs], dim=1)


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.col = nn.Sequential(
            nn.Conv2d(cin + 2, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 48, 1), nn.ReLU())
        self.c1 = nn.Conv2d(48, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)

    def forward(self, obs):
        c = self.col(add_coords(obs))
        x = F.relu(self.c1(c))
        return F.relu(x + self.c2(x))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        T = int(spec.n_types)
        self.trunk = Trunk(int(spec.in_channels))
        self.head = nn.Conv2d(W, H, 3, padding=1)
        self.mid = nn.Conv2d(H, H, 1)
        self.out = nn.Conv2d(H, T, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = self.trunk(obs)
        h = F.relu(self.head(f))
        h = F.relu(self.mid(h))
        place = self.out(h).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
