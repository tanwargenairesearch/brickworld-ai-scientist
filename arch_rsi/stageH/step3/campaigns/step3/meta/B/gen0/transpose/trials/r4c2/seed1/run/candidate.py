import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

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
        cin = int(spec.in_channels) + 4 * self.T + 2
        self.col1 = nn.Conv2d(cin, COLH, 1)
        self.col2 = nn.Conv2d(COLH, W, 1)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.h1 = nn.Conv2d(W, HH, 3, padding=1)
        self.h_skip = nn.Conv2d(W, HH, 1)
        self.outs = nn.ModuleList([nn.Conv2d(HH, self.T, 1) for _ in range(4)])
        self.glob = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T = self.Y, self.X, self.T
        mp = mask[:, :self.n_place].to(obs.dtype)
        mk1 = mp.reshape(B, T, Y, X)
        mk2 = mp.reshape(B, T, X, Y).transpose(2, 3)
        mk3 = mk1.flip(2)
        mk4 = mk1.flip(3)
        yy = torch.linspace(0.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(0.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        inp = torch.cat([obs, mk1, mk2, mk3, mk4, yy, xx], dim=1)
        s = F.relu(self.col1(inp))
        s = F.relu(self.col2(s))
        x = F.relu(s + self.c1(s))
        f = F.relu(x + self.c2(x))
        h = F.relu(self.h1(f) + self.h_skip(f))
        m0 = self.outs[0](h)
        m1 = self.outs[1](h)
        m2 = self.outs[2](h)
        m3 = self.outs[3](h)
        place = (m0.flatten(1) + m1.transpose(2, 3).flatten(1)
                 + m2.flip(2).flatten(1) + m3.flip(3).flatten(1))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
