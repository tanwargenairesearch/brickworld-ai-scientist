import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0015, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.98, "gae_lambda": 0.95}

STEM = 64
WIDTH = 40
HEAD = 48


def tf(m, k):
    if k >= 4:
        m = m.transpose(2, 3)
    r = k % 4
    if r == 1 or r == 3:
        m = torch.flip(m, [3])
    if r >= 2:
        m = torch.flip(m, [2])
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.ntf = 8 if int(spec.Y) == int(spec.X) else 4
        self.nh = 2 * self.ntf
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.b1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.b2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.ctx = nn.Linear(2 * WIDTH, WIDTH)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h2 = nn.Conv2d(HEAD, self.nh * self.T, 1)
        nn.init.normal_(self.h2.weight, std=0.01)
        nn.init.zeros_(self.h2.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = self.stem(obs)
        f = F.relu(f + self.b1(f))
        c = self.ctx(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        f = F.relu(f + c[:, :, None, None])
        f = F.relu(f + self.b2(f))
        o = self.h2(F.relu(self.h1(f)))
        B = o.shape[0]
        Y = o.shape[2]
        X = o.shape[3]
        o = o.view(B, self.nh, self.T, Y, X)
        place = None
        for k in range(self.ntf):
            m1 = tf(o[:, k], k).flatten(1)
            m2 = tf(o[:, self.ntf + k], k).permute(0, 2, 3, 1).flatten(1)
            s = m1 + m2
            place = s if place is None else place + s
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
