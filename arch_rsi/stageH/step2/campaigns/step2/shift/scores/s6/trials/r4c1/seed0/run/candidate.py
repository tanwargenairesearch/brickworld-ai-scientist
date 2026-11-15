import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "lr": 0.0015, "ent_coef": 0.003, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
WIDTH = 48
PROJ = 8
HID = 32


def views(p, square):
    vs = [p, p.flip(2), p.flip(3), p.flip(2).flip(3)]
    if square:
        vs = vs + [v.transpose(2, 3) for v in vs]
    return torch.cat(vs, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        self.square = int(spec.X) == int(spec.Y)
        nv = 8 if self.square else 4
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.h1 = nn.Conv2d(nv * PROJ + self.nt, HID, 3, padding=1)
        self.h2 = nn.Conv2d(HID, HID, 3, padding=1)
        self.out = nn.Conv2d(HID, self.nt, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        m = mask[:, :self.nt * Y * X].float().reshape(B, self.nt, Y, X)
        s = self.stem(obs)
        x = F.relu(s + self.c1(s))
        f = F.relu(x + self.c2(x))
        p = F.relu(self.proj(f))
        h = torch.cat([views(p, self.square), m], dim=1)
        h = F.relu(self.h1(h))
        h = F.relu(h + self.h2(h))
        place = self.out(h).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
