import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "lr": 0.001, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.003}

W = 32
NB = 3
HH = 64


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a = nn.Conv2d(c, c, 3, padding=1)
        self.b = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.b(F.relu(self.a(x))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.ky = max(a for a, b in fps)
        self.kx = max(b for a, b in fps)
        T = int(spec.n_types)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + 2, 64, 1), nn.ReLU(),
            nn.Conv2d(64, W, 1), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(W) for _ in range(NB)])
        self.win = nn.Conv2d(W, HH, (self.ky, self.kx))
        self.out = nn.Conv2d(HH, T, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        s = self.stem(torch.cat([obs, yy, xx], dim=1))
        f = self.blocks(s)
        q = F.pad(f, (0, self.kx - 1, 0, self.ky - 1))
        h = F.relu(self.win(q))
        place = self.out(h).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
