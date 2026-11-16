import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 64, "n_minibatches": 16, "lr": 0.001, "ent_coef": 0.001, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}

STEM = 64
WIDTH = 32
BLOCKS = 4
HID = 64


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.glob = nn.Linear(2 * c, c)

    def forward(self, x):
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        return F.relu(x + self.conv(x) + g[:, :, None, None])


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.blocks = nn.ModuleList([Block(WIDTH) for _ in range(BLOCKS)])
        self.foot = nn.Conv2d(WIDTH, WIDTH, 2)
        self.place = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, HID), nn.ReLU())
        self.done_head = nn.Linear(HID, 1)
        self.value_head = nn.Linear(HID, 1)
        nn.init.orthogonal_(self.place.weight, 0.01)
        nn.init.zeros_(self.place.bias)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)
        nn.init.orthogonal_(self.value_head.weight, 1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs, mask):
        x = self.stem(obs)
        for b in self.blocks:
            x = b(x)
        h = F.relu(self.foot(F.pad(x, (0, 1, 0, 1))))
        place = self.place(h).flatten(1)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
