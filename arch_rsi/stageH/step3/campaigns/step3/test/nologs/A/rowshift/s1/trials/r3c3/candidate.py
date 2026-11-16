import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_minibatches": 16, "gamma": 0.98, "gae_lambda": 0.95, "lr": 0.001, "ent_coef": 0.002}

W = 32
HID = 64


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.c2(F.relu(self.c1(x))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.kh = max(int(a) for a, b in spec.footprints)
        self.kw = max(int(b) for a, b in spec.footprints)
        cin = spec.in_channels + self.T
        self.stem = nn.Sequential(
            nn.Conv2d(cin, 64, 1), nn.ReLU(),
            nn.Conv2d(64, W, 1), nn.ReLU())
        self.b1 = Block(W)
        self.glob1 = nn.Linear(2 * W, W)
        self.b2 = Block(W)
        self.foot = nn.Conv2d(W, W, (self.kh, self.kw))
        self.out = nn.Conv2d(W, self.T, 1)
        self.g = nn.Sequential(nn.Linear(2 * W, HID), nn.ReLU())
        self.done = nn.Linear(HID, 1)
        self.value = nn.Linear(HID, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.n_place].reshape(B, self.T, self.Y, self.X).to(obs.dtype)
        x = torch.cat([obs, m], dim=1)
        h = self.stem(x)
        h = self.b1(h)
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        h = F.relu(h + self.glob1(pooled)[:, :, None, None])
        h = self.b2(h)
        q = F.pad(h, (0, self.kw - 1, 0, self.kh - 1))
        p = F.relu(self.foot(q))
        place = self.out(p).flatten(1)
        g = self.g(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done(g)], dim=1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
