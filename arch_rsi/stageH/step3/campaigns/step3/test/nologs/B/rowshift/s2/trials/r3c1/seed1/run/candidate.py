import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "ent_coef": 0.03, "gamma": 0.98, "gae_lambda": 0.95}

WIDTH = 32
NBLOCKS = 2
HID = 32
GHID = 64


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.GroupNorm(4, c)
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(4, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.relu(self.n1(x)))
        h = self.c2(F.relu(self.n2(h)))
        return x + h


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = spec.n_types
        self.Y = spec.Y
        self.X = spec.X
        self.n_place = spec.n_place
        cin = spec.in_channels + spec.n_types
        self.stem = nn.Conv2d(cin, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([Block(WIDTH) for _ in range(NBLOCKS)])
        self.out_norm = nn.GroupNorm(4, WIDTH)
        self.ph = nn.Conv2d(WIDTH, HID, 1)
        self.po = nn.Conv2d(HID, spec.n_types, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GHID), nn.ReLU())
        self.done_head = nn.Linear(GHID, 1)
        self.vh = nn.Sequential(nn.Linear(GHID, GHID), nn.ReLU())
        self.value_head = nn.Linear(GHID, 1)
        nn.init.orthogonal_(self.po.weight, 0.01)
        nn.init.zeros_(self.po.bias)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.T, self.Y, self.X)
        x = self.stem(torch.cat([obs, m], dim=1))
        for b in self.blocks:
            x = b(x)
        f = F.relu(self.out_norm(x))
        h = F.relu(self.ph(f))
        place = self.po(h).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        done = self.done_head(g)
        value = self.value_head(self.vh(g)).squeeze(1)
        return torch.cat([place, done], dim=1), value


def build_policy(spec):
    return Net(spec)
