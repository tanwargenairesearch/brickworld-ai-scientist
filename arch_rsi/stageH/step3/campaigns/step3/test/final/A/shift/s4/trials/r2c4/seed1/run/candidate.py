import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32,
          "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

WIDTH = 32
SKIP = 16
HID = 40
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
        self.T = spec.n_types
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        K = max(max(a, b) for a, b in fps)
        self.trunk = Trunk(spec)
        self.hid = nn.Conv2d(self.trunk.out_ch, HID, 2 * K - 1, padding=K - 1)
        self.out = nn.Conv2d(HID, self.T, 1)
        self.glob = nn.Linear(2 * WIDTH, GHID)
        self.done_head = nn.Linear(GHID, 1)
        self.value_head = nn.Linear(GHID, 1)
        self.dense = nn.Linear(GHID, spec.n_actions)
        self.remap = nn.Linear(spec.n_actions, spec.n_actions, bias=False)
        init_layer(self.hid, math.sqrt(2))
        init_layer(self.out, 0.01)
        init_layer(self.glob, math.sqrt(2))
        init_layer(self.done_head, 0.01)
        init_layer(self.dense, 0.01)
        init_layer(self.remap, 1.0)
        init_layer(self.value_head, 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        f, x = self.trunk(obs)
        M = self.out(F.relu(self.hid(f)))
        g = F.relu(self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)))
        z = torch.cat([M.reshape(B, -1), self.done_head(g)], dim=1)
        logits = self.remap(z) + self.dense(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
