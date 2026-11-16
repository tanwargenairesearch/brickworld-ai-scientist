import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "lr": 0.001, "ent_coef": 0.003}

STEM = 64
W = 32
FLAT = 8
HID = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + 2, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.red = nn.Conv2d(W, FLAT, 1)
        self.fc = nn.Linear(FLAT * Y * X + 2 * W, HID)
        self.pi = nn.Linear(HID, spec.n_actions)
        self.v = nn.Linear(HID, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        x = torch.cat([obs, ys, xs], dim=1)
        s = self.stem(x)
        h = F.relu(s + self.c1(s))
        f = F.relu(h + self.c2(h))
        r = F.relu(self.red(f)).flatten(1)
        g = torch.cat([r, f.mean((2, 3)), f.amax((2, 3))], dim=1)
        z = F.relu(self.fc(g))
        return self.pi(z), self.v(z).squeeze(1)


def build_policy(spec):
    return Net(spec)
