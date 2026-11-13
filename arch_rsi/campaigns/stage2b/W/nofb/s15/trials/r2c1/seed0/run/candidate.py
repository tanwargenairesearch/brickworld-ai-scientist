import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
DEPTH = 3

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(DEPTH - 1)])
        self.ctx = nn.Linear(2 * WIDTH, WIDTH)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 2)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        for i, conv in enumerate(self.blocks):
            if i == 0:
                c = self.ctx(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
                f = f + c[:, :, None, None]
            f = f + F.relu(conv(f))
        place = self.place_head(F.pad(f, (0, 1, 0, 1))).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
