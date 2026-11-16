import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.02, "lr": 0.001}

STEM_H = 64
W = 40
HID = 32


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


class FootHead(nn.Module):
    def __init__(self, c, footprints, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.convs = nn.ModuleList([nn.Conv2d(c, hid, (dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)

    def forward(self, f):
        res = []
        for (dy, dx), conv, out in zip(self.fps, self.convs, self.outs):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            h = F.relu(conv(q))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem1 = nn.Conv2d(spec.in_channels, STEM_H, 1)
        self.stem2 = nn.Conv2d(STEM_H, W, 1)
        self.b1 = Block(W)
        self.b2 = Block(W)
        self.norm = nn.GroupNorm(4, W)
        self.head = FootHead(W, spec.footprints, HID)
        self.done_mlp = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_out = nn.Linear(64, 1)
        nn.init.normal_(self.done_out.weight, std=0.01)
        nn.init.zeros_(self.done_out.bias)
        self.val_mlp = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.val_out = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem2(F.relu(self.stem1(obs)))
        f = self.b2(self.b1(s))
        f = F.relu(self.norm(f))
        place = self.head(f)
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        done = self.done_out(self.done_mlp(g))
        value = self.val_out(self.val_mlp(g)).squeeze(1)
        logits = torch.cat([place, done], dim=1)
        return logits, value


def build_policy(spec):
    return Net(spec)
