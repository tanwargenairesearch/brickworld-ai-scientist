import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0015, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 1, "ent_coef": 0.01, "gamma": 0.97, "gae_lambda": 0.92}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 16
HID = 24


class WindowHead(nn.Module):
    def __init__(self, c, n_types):
        super().__init__()
        self.proj = nn.Conv2d(c, PROJ, 1)
        self.conv = nn.Conv2d(2 * PROJ, n_types * HID, 3, padding=1)
        self.out = nn.Conv2d(n_types * HID, n_types, 1, groups=n_types)

    def forward(self, f):
        p = F.relu(self.proj(f))
        if p.shape[2] == p.shape[3]:
            pt = p.transpose(2, 3)
        else:
            pt = torch.zeros_like(p)
        q = torch.cat([p, pt], dim=1)
        h = F.relu(self.conv(q))
        return self.out(h).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.NP = int(spec.n_place)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + self.T, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = WindowHead(WIDTH, self.T)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        m = mask[:, :self.NP].to(obs.dtype).reshape(B, self.T, Y, X)
        inp = torch.cat([obs, m], dim=1)
        s = self.stem(inp)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
