import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 1e-3, "n_envs": 64, "rollout_len": 64, "n_minibatches": 32, "ppo_epochs": 1, "ent_coef": 0.01, "gamma": 0.9, "gae_lambda": 0.85}

W = 48
HP = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.NP = int(spec.n_place)
        cin = int(spec.in_channels) + self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, 64, 1), nn.ReLU(),
            nn.Conv2d(64, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=2, dilation=2)
        self.gl = nn.Linear(2 * W, W)
        self.hproj = nn.Conv2d(W, HP, 1)
        self.hout = nn.Conv2d(HP, self.T, 5, padding=2)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        ys = torch.linspace(-1.0, 1.0, self.Y).view(1, 1, self.Y, 1).expand(1, 1, self.Y, self.X)
        xs = torch.linspace(-1.0, 1.0, self.X).view(1, 1, 1, self.X).expand(1, 1, self.Y, self.X)
        self.register_buffer('coords', torch.cat([ys, xs], dim=1).contiguous())

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.NP].to(obs.dtype).reshape(B, self.T, self.Y, self.X)
        c = self.coords.expand(B, -1, -1, -1)
        x = self.stem(torch.cat([obs, m, c], dim=1))
        x = F.relu(x + self.c1(x))
        x = F.relu(x + self.c2(x))
        gf = torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)
        x = F.relu(x + self.gl(gf)[:, :, None, None])
        h = F.relu(self.hproj(x))
        place = self.hout(h).flatten(1)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
