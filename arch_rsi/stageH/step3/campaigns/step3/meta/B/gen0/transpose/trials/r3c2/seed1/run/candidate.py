import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0005, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 3, "ent_coef": 0.04, "gamma": 0.99, "gae_lambda": 0.95}

COLH = 64
W = 40
HH = 32
GH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        cin = int(spec.in_channels) + self.T + 2
        self.col1 = nn.Conv2d(cin, COLH, 1)
        self.col2 = nn.Conv2d(COLH, W, 1)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.h1 = nn.Conv2d(W, HH, 3, padding=1)
        self.h_skip = nn.Conv2d(W, HH, 1)
        self.out = nn.Conv2d(HH, self.T, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)
        self.done_w = nn.Parameter(torch.zeros(1))

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T = self.Y, self.X, self.T
        mk = mask[:, :self.n_place].to(obs.dtype).reshape(B, T, Y, X)
        yy = torch.linspace(0.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(0.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        inp = torch.cat([obs, mk, yy, xx], dim=1)
        s = F.relu(self.col1(inp))
        s = F.relu(self.col2(s))
        x = F.relu(s + self.c1(s))
        f = F.relu(x + self.c2(x))
        h = F.relu(self.h1(f) + self.h_skip(f))
        place = self.out(h).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        done = self.done_head(g) + self.done_w * torch.logsumexp(place, dim=1, keepdim=True)
        logits = torch.cat([place, done], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
