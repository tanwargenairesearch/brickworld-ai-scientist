import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
W = 48
FL = 16
HID = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.np = int(spec.n_place)
        self.T = self.np // (self.Y * self.X)
        cin = int(spec.in_channels) + self.T
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.red = nn.Conv2d(2 * W, FL, 1)
        n = FL * self.Y * self.X
        self.fc = nn.Linear(n, HID)
        self.pi_direct = nn.Linear(n, int(spec.n_actions))
        self.pi_hid = nn.Linear(HID, int(spec.n_actions))
        self.v = nn.Linear(HID, 1)
        nn.init.orthogonal_(self.pi_direct.weight, 0.01)
        nn.init.zeros_(self.pi_direct.bias)
        nn.init.orthogonal_(self.pi_hid.weight, 0.01)
        nn.init.zeros_(self.pi_hid.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.np].to(obs.dtype).reshape(B, self.T, self.Y, self.X)
        s = self.stem(torch.cat([obs, m], 1))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        r = F.relu(self.red(torch.cat([s, f], 1)))
        flat = r.flatten(1)
        h = F.relu(self.fc(flat))
        logits = self.pi_direct(flat) + self.pi_hid(h)
        return logits, self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
