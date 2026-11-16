import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32,
          "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

WIDTH = 32
SKIP = 16
D = 16
GHID = 64


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain=gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class Trunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.stem1 = nn.Conv2d(c, 64, 1)
        self.stem2 = nn.Conv2d(64, 48, 1)
        self.c1 = nn.Conv2d(48, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.out_ch = WIDTH + SKIP
        for m in [self.stem1, self.stem2, self.c1, self.c2, self.skip]:
            init_layer(m, math.sqrt(2))

    def forward(self, obs):
        s = F.relu(self.stem2(F.relu(self.stem1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        sk = F.relu(self.skip(obs))
        return torch.cat([x, sk], dim=1), x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec)
        self.red = nn.Conv2d(self.trunk.out_ch, D, 1)
        self.glob = nn.Linear(2 * WIDTH, GHID)
        self.policy = nn.Linear(D * spec.Y * spec.X + GHID, spec.n_actions)
        self.value_head = nn.Linear(GHID, 1)
        init_layer(self.red, math.sqrt(2))
        init_layer(self.glob, math.sqrt(2))
        init_layer(self.policy, 0.01)
        init_layer(self.value_head, 1.0)

    def forward(self, obs, mask):
        f, x = self.trunk(obs)
        e = F.relu(self.red(f)).flatten(1)
        g = F.relu(self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)))
        logits = self.policy(torch.cat([e, g], dim=1))
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
