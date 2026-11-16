import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32, "lr": 0.0007, "ent_coef": 0.002, "gamma": 0.97, "gae_lambda": 0.95}

W = 40
NB = 3
GH = 64


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.film = nn.Linear(2 * c, c)

    def forward(self, x):
        g = torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)
        b = self.film(g).unsqueeze(2).unsqueeze(3)
        return F.relu(x + self.conv(x) + b)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.np = int(spec.n_place)
        self.sq = self.Y == self.X
        self.stem = nn.Conv2d(int(spec.in_channels) + 2 * self.T, W, 1)
        self.blocks = nn.ModuleList([Block(W) for _ in range(NB)])
        self.out = nn.Conv2d(W, 2 * self.T, 3, padding=1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(GH, 1)

    def swap(self, t):
        if self.sq:
            return t.transpose(2, 3)
        return torch.flip(t, dims=[2, 3])

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.np].to(obs.dtype).reshape(B, self.T, self.Y, self.X)
        x = torch.cat([obs, m, self.swap(m)], dim=1)
        f = F.relu(self.stem(x))
        for blk in self.blocks:
            f = blk(f)
        o = self.out(f)
        a = o[:, :self.T]
        b = self.swap(o[:, self.T:])
        place = (a + b).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
