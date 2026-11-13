import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 32, "n_minibatches": 16, "lr": 0.0007, "ent_coef": 0.005, "gamma": 0.97, "gae_lambda": 0.95}

WIDTH = 64
DEPTH = 3


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.convs = nn.ModuleList()
        c_in = spec.in_channels
        for _ in range(DEPTH):
            self.convs.append(nn.Conv2d(c_in, WIDTH, 3, padding=1))
            c_in = WIDTH
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.ctx = nn.Linear(WIDTH, WIDTH)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 3, padding=1)
        nn.init.normal_(self.place_head.weight, std=0.01)
        nn.init.zeros_(self.place_head.bias)
        self.done_head = nn.Linear(WIDTH, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = obs
        for conv in self.convs:
            f = F.relu(conv(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = F.relu(f + self.ctx(g)[:, :, None, None])
        place = self.place_head(h).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
