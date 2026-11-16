import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 512, "rollout_len": 16, "n_minibatches": 32, "lr": 0.0007, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.01}

STEM = 64
W = 48
H = 24
G = 64


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n = nn.GroupNorm(4, c)
        self.c = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c(F.relu(self.n(x)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1))
        self.b1 = Block(W)
        self.b2 = Block(W)
        self.norm = nn.GroupNorm(4, W)
        self.hid = nn.Conv2d(W, H, 3, padding=1)
        self.variants = []
        for tr in (False, True):
            for fy in (False, True):
                for fx in (False, True):
                    for cm in (False, True):
                        self.variants.append((tr, fy, fx, cm))
        self.outs = nn.ModuleList([nn.Conv2d(H, self.T, 3, padding=1) for _ in self.variants])
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)

    def forward(self, obs, mask):
        x = self.stem(obs)
        x = self.b2(self.b1(x))
        f = F.relu(self.norm(x))
        h = F.relu(self.hid(f))
        place = None
        for (tr, fy, fx, cm), out in zip(self.variants, self.outs):
            o = out(h)
            if tr:
                o = o.transpose(2, 3)
            dims = []
            if fy:
                dims.append(2)
            if fx:
                dims.append(3)
            if dims:
                o = torch.flip(o, dims)
            if cm:
                o = o.permute(0, 2, 3, 1)
            o = o.reshape(o.shape[0], -1)
            place = o if place is None else place + o
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
