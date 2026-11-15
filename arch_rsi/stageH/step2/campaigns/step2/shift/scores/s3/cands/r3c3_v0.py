import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "ppo_epochs": 1, "ent_coef": 0.003}

W = 32
RED = 16
HID = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        Y = int(spec.Y)
        X = int(spec.X)
        cin = int(spec.in_channels) + self.T + 2
        self.stem = nn.Sequential(nn.Conv2d(cin, 64, 1), nn.ReLU(), nn.Conv2d(64, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.red = nn.Conv2d(W, RED, 1)
        self.fc = nn.Linear(RED * Y * X + 2 * W, HID)
        self.pi = nn.Linear(HID, int(spec.n_actions))
        self.v = nn.Linear(HID, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        T = self.T
        m = mask[:, :T * Y * X].reshape(B, T, Y, X).to(obs.dtype)
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        x = self.stem(torch.cat([obs, m, yy, xx], dim=1))
        x = F.relu(x + self.c1(x))
        x = F.relu(x + self.c2(x))
        r = F.relu(self.red(x)).flatten(1)
        h = F.relu(self.fc(torch.cat([r, x.mean((2, 3)), x.amax((2, 3))], dim=1)))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
