import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 32, "n_minibatches": 16, "ent_coef": 0.005, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
W = 40
K = 5
R = 8
H = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.np = int(spec.n_place)
        self.sq = self.Y == self.X
        self.nv = 5 if self.sq else 4
        cin = int(spec.in_channels) + self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.out = nn.Conv2d(W, self.T * self.nv, K, padding=K // 2)
        self.red = nn.Conv2d(W, R, 1)
        self.fc = nn.Linear(R * self.Y * self.X + 2 * W, H)
        self.dense_out = nn.Linear(H, int(spec.n_actions))
        self.value_head = nn.Linear(H, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = self.Y
        X = self.X
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        m = mask[:, :self.np].reshape(B, self.T, Y, X).to(obs.dtype)
        s = self.stem(torch.cat([obs, m, ys, xs], dim=1))
        x = F.relu(s + self.c1(s))
        f = F.relu(x + self.c2(x))
        o = self.out(f).reshape(B, self.nv, self.T, Y, X)
        L = o[:, 0] + o[:, 1].flip(2) + o[:, 2].flip(3) + o[:, 3].flip(2).flip(3)
        if self.sq:
            L = L + o[:, 4].transpose(2, 3)
        local = L.flatten(1)
        flat = F.relu(self.red(f)).flatten(1)
        h = F.relu(self.fc(torch.cat([flat, f.mean((2, 3)), f.amax((2, 3))], dim=1)))
        dense = self.dense_out(h)
        logits = dense + torch.cat([local, torch.zeros_like(local[:, :1])], dim=1)
        return logits, self.value_head(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
