import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 64, "rollout_len": 32, "n_minibatches": 32}

W = 40
HH = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.n_types = int(spec.n_types)
        self.stem = nn.Sequential(nn.Conv2d(spec.in_channels, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.gctx = nn.Linear(2 * W, W)
        self.c3 = nn.Conv2d(W, W, 3, padding=1)
        self.c4 = nn.Conv2d(W, W, 3, padding=1)
        self.h1 = nn.Conv2d(W, HH, 3, padding=1)
        self.h2 = nn.Conv2d(HH, self.n_types, 1)
        nn.init.normal_(self.h2.weight, std=0.01)
        nn.init.zeros_(self.h2.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(s + self.c1(s))
        x = F.relu(x + self.c2(x))
        g0 = torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)
        x = F.relu(x + self.gctx(g0)[:, :, None, None])
        x = F.relu(x + self.c3(x))
        f = F.relu(x + self.c4(x))
        place = self.h2(F.relu(self.h1(f))).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
