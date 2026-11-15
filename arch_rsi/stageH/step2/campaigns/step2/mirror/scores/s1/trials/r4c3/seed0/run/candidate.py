import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.001, "ent_coef": 0.003, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
HEADC = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        self.npl = int(spec.n_place)
        cin = spec.in_channels + 4 * self.nt
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = nn.Sequential(
            nn.Conv2d(WIDTH, HEADC, 3, padding=1), nn.ReLU(),
            nn.Conv2d(HEADC, 4 * self.nt, 1))
        self.cnt = nn.Conv2d(WIDTH, 8, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + 8, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        nt = self.nt
        m = mask[:, :self.npl].float()
        m0 = m.view(B, nt, Y, X)
        m1 = m.view(B, Y, X, nt).permute(0, 3, 1, 2)
        m2 = m.view(B, nt, X, Y).transpose(2, 3)
        m3 = m.view(B, X, Y, nt).permute(0, 3, 2, 1)
        s = self.stem(torch.cat([obs, m0, m1, m2, m3], dim=1))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        h = self.head(f)
        p0 = h[:, 0:nt].flatten(1)
        p1 = h[:, nt:2 * nt].permute(0, 2, 3, 1).flatten(1)
        p2 = h[:, 2 * nt:3 * nt].transpose(2, 3).flatten(1)
        p3 = h[:, 3 * nt:4 * nt].permute(0, 3, 2, 1).flatten(1)
        place = p0 + p1 + p2 + p3
        c = F.relu(self.cnt(f)).sum((2, 3)) / 8.0
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), c], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
