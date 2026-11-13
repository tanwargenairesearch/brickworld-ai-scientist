import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 40
N_RES = 3
RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 3, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        W = WIDTH
        self.stem = nn.Conv2d(spec.in_channels, W, 3, padding=1)
        self.convs = nn.ModuleList([nn.Conv2d(W, W, 3, padding=1) for _ in range(N_RES)])
        self.glob_in = nn.Linear(2 * W, W)
        self.glob_out = nn.Linear(W, W)
        self.head_hid = nn.Conv2d(W, 32, 1)
        self.place_head = nn.Conv2d(32, spec.n_types, 1)
        nn.init.normal_(self.place_head.weight, std=0.01)
        nn.init.zeros_(self.place_head.bias)
        self.trunk = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        h = F.relu(self.stem(obs))
        for i, conv in enumerate(self.convs):
            h = h + F.relu(conv(h))
            if i == 0:
                g = F.relu(self.glob_in(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)))
                h = F.relu(h + self.glob_out(g)[:, :, None, None])
        place = self.place_head(F.relu(self.head_hid(h))).flatten(1)
        t = self.trunk(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(t)], dim=1)
        return logits, self.value_head(t).squeeze(1)


def build_policy(spec):
    return Net(spec)
