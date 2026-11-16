import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 32, "n_minibatches": 16, "ppo_epochs": 1, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95}

W = 48


class Block(nn.Module):
    def __init__(self, c, dil):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=dil, dilation=dil)
        self.g = nn.Linear(c, c)

    def forward(self, x):
        g = self.g(x.mean((2, 3)))[:, :, None, None]
        return F.relu(x + self.conv(x) + g)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.X = int(spec.X)
        self.Y = int(spec.Y)
        self.nt = int(spec.n_types)
        self.n_place = int(spec.n_place)
        Y, X = self.Y, self.X
        cy = torch.linspace(-1.0, 1.0, Y).reshape(1, 1, Y, 1).expand(1, 1, Y, X)
        cx = torch.linspace(-1.0, 1.0, X).reshape(1, 1, 1, X).expand(1, 1, Y, X)
        self.register_buffer("coords", torch.cat([cy, cx], dim=1).contiguous())
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + self.nt + 2, 64, 1), nn.ReLU(),
            nn.Conv2d(64, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.blocks = nn.Sequential(Block(W, 1), Block(W, 2), Block(W, 1))
        self.head = nn.Sequential(nn.Conv2d(W, 32, 1), nn.ReLU(), nn.Conv2d(32, self.nt, 3, padding=1))
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.nt, self.Y, self.X)
        c = self.coords.expand(B, -1, -1, -1)
        x = self.stem(torch.cat([obs, m, c], dim=1))
        x = F.relu(self.c1(x))
        f = self.blocks(x)
        place = self.head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
