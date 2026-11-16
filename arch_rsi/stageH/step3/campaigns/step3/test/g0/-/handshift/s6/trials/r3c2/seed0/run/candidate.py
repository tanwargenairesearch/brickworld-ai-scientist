import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "ent_coef": 0.03, "gamma": 0.99, "gae_lambda": 0.95}

W = 32
HH = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.A = int(spec.n_place)
        cin = int(spec.in_channels) + self.T
        self.stem = nn.Sequential(nn.Conv2d(cin, 48, 1), nn.ReLU(), nn.Conv2d(48, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.gctx = nn.Linear(2 * W, W)
        self.c3 = nn.Conv2d(W, W, 3, padding=1)
        self.h1 = nn.Conv2d(W, HH, 3, padding=1)
        self.head_a = nn.Conv2d(HH, self.T, 5, padding=2)
        self.head_b = nn.Conv2d(HH, self.T, 5, padding=2)
        for h in (self.head_a, self.head_b):
            nn.init.normal_(h.weight, std=0.01)
            nn.init.zeros_(h.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.A].to(obs.dtype).reshape(B, self.T, self.Y, self.X)
        s = self.stem(torch.cat([obs, m], dim=1))
        x = F.relu(s + self.c1(s))
        x = F.relu(x + self.c2(x))
        g0 = torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)
        x = F.relu(x + self.gctx(g0)[:, :, None, None])
        f = F.relu(x + self.c3(x))
        h = F.relu(self.h1(f))
        pa = self.head_a(h).flatten(1)
        pb = self.head_b(h.transpose(2, 3).contiguous()).flatten(1)
        place = pa + pb
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
