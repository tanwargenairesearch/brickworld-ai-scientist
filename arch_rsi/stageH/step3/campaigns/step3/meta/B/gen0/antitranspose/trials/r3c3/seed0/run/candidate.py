import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
U = 32
H1 = 48
H2 = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.ks = [max(a, b) for a, b in self.fps]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.u = nn.Conv2d(COL + WIDTH, U, 1)
        self.k1 = nn.ModuleList([nn.Conv2d(U, H1, k) for k in self.ks])
        self.k2 = nn.ModuleList([nn.Conv2d(H1, H2, 1) for _ in self.ks])
        self.k3 = nn.ModuleList([nn.Conv2d(H2, 1, 1) for _ in self.ks])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + U, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        u = F.relu(self.u(torch.cat([s, f], 1)))
        res = []
        for k, a, b, c in zip(self.ks, self.k1, self.k2, self.k3):
            q = F.pad(u, (0, k - 1, 0, k - 1)) if k > 1 else u
            h = F.relu(a(q))
            h = F.relu(b(h))
            res.append(c(h).flatten(1))
        place = torch.cat(res, 1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), u.amax((2, 3))], 1))
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
