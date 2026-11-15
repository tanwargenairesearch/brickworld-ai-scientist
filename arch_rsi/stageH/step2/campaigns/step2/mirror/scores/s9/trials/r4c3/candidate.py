import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 3, "lr": 0.0007, "gamma": 0.95, "gae_lambda": 0.9, "ent_coef": 0.003}

STEM = 64
W = 48
DEPTH = 3
FLAT = 16
G = 64
ME = 32
HID = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = int(spec.Y) * int(spec.X)
        na = int(spec.n_actions)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.blocks = nn.ModuleList([nn.Conv2d(W, W, 3, padding=1) for _ in range(DEPTH)])
        self.proj = nn.Conv2d(W, FLAT, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU())
        self.memb = nn.Linear(na, ME)
        self.fc = nn.Linear(FLAT * n + G + ME, HID)
        self.out = nn.Linear(HID, na)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.value_head = nn.Sequential(nn.Linear(G + HID, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, obs, mask):
        x = self.stem(obs)
        for c in self.blocks:
            x = F.relu(x + c(x))
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        p = F.relu(self.proj(x)).flatten(1)
        me = F.relu(self.memb(mask.to(obs.dtype)))
        h = F.relu(self.fc(torch.cat([p, g, me], dim=1)))
        logits = self.out(h)
        value = self.value_head(torch.cat([g, h], dim=1)).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
