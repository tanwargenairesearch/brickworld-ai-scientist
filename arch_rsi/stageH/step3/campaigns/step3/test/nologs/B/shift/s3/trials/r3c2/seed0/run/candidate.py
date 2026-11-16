import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 40
DEPTH = 3
HEAD = 48
GLOB = 64

RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv(F.relu(self.norm(x)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 1)
        self.blocks = nn.Sequential(*[Block(WIDTH) for _ in range(DEPTH)])
        self.norm = nn.GroupNorm(8, WIDTH)
        self.place_hid = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.place_out = nn.Conv2d(HEAD, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        nn.init.normal_(self.place_out.weight, std=0.01)
        nn.init.zeros_(self.place_out.bias)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)

    def forward(self, obs, mask):
        x = self.blocks(self.stem(obs))
        f = F.relu(self.norm(x))
        h = F.relu(self.place_hid(f))
        place = self.place_out(h).transpose(2, 3).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
