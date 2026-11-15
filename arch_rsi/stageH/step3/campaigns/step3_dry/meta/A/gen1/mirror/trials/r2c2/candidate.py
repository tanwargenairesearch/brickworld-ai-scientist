import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "gamma": 0.99, "gae_lambda": 0.95}

W = 40
DS = 16
DONE_BIAS = 3.0


class Block(nn.Module):
    def __init__(self, c, dil):
        super().__init__()
        self.a = nn.Conv2d(c, c, 3, padding=dil, dilation=dil)
        self.b = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.b(F.relu(self.a(x))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.Y = spec.Y
        self.X = spec.X
        self.nt = spec.n_types
        self.n_place = spec.n_place
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + spec.n_types, 64, 1), nn.ReLU(),
            nn.Conv2d(64, W, 1), nn.ReLU())
        self.blocks = nn.Sequential(Block(W, 1), Block(W, 2), Block(W, 1))
        self.head = nn.Conv2d(W, spec.n_types, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.dconv = nn.Conv2d(2 * spec.Z, DS, 1)
        self.done_head = nn.Linear(64 + 2 * DS, 1)
        self.done_bias = nn.Parameter(torch.full((1,), DONE_BIAS))
        self.value_head = nn.Linear(64 + 2 * DS, 1)
        nn.init.zeros_(self.done_head.weight)
        nn.init.zeros_(self.done_head.bias)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Z = self.Z
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.nt, self.Y, self.X)
        f = self.blocks(self.stem(torch.cat([obs, m], dim=1)))
        place = self.head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        d = F.relu(self.dconv(obs[:, 2 * Z:4 * Z]))
        gd = torch.cat([g, d.amax((2, 3)), d.mean((2, 3))], dim=1)
        done = self.done_head(gd) + self.done_bias
        logits = torch.cat([place, done], dim=1)
        return logits, self.value_head(gd).squeeze(1)


def build_policy(spec):
    return Net(spec)
