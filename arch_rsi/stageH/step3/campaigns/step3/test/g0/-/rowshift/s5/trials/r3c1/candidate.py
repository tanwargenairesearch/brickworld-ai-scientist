import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ent_coef": 0.005, "gamma": 0.97}

C1 = 64
W = 48
R = 8
H = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + 2, C1, 1), nn.ReLU(),
            nn.Conv2d(C1, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.local = nn.Conv2d(W, spec.n_types, 1)
        self.red = nn.Conv2d(W, R, 1)
        self.fc = nn.Linear(R * self.Y * self.X + 2 * W, H)
        self.dense_out = nn.Linear(H, spec.n_actions)
        self.value_head = nn.Linear(H, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        ys = torch.linspace(-1.0, 1.0, self.Y, device=obs.device, dtype=obs.dtype).view(1, 1, self.Y, 1).expand(B, 1, self.Y, self.X)
        xs = torch.linspace(-1.0, 1.0, self.X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, self.X).expand(B, 1, self.Y, self.X)
        s = self.stem(torch.cat([obs, ys, xs], dim=1))
        x = F.relu(s + self.c1(s))
        f = F.relu(x + self.c2(x))
        local = self.local(f).flatten(1)
        flat = F.relu(self.red(f)).flatten(1)
        g = torch.cat([flat, f.mean((2, 3)), f.amax((2, 3))], dim=1)
        h = F.relu(self.fc(g))
        dense = self.dense_out(h)
        logits = dense + torch.cat([local, torch.zeros_like(local[:, :1])], dim=1)
        return logits, self.value_head(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
