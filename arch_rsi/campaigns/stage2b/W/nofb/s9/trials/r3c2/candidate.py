import torch
import torch.nn as nn

WIDTH = 48
GW = 64
RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.conv3 = nn.Conv2d(WIDTH, WIDTH, 3, padding=2, dilation=2)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GW), nn.ReLU())
        self.film = nn.Linear(GW, WIDTH)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.done_head = nn.Linear(GW, 1)
        self.value_head = nn.Linear(GW, 1)

    def forward(self, obs, mask):
        f = torch.relu(self.stem(obs))
        f = f + torch.relu(self.conv2(f))
        f = f + torch.relu(self.conv3(f))
        q = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = torch.relu(f + self.film(q)[:, :, None, None])
        place = self.place_head(h).flatten(1)
        logits = torch.cat([place, self.done_head(q)], dim=1)
        return logits, self.value_head(q).squeeze(1)


def build_policy(spec):
    return Net(spec)
