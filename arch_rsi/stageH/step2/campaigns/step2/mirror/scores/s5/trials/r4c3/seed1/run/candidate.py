import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "lr": 0.001, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.03, "clip_eps": 0.3}

STEM = 64
W = 40
GH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.convs = nn.ModuleList([
            nn.Conv2d(W, W, 3, padding=1),
            nn.Conv2d(W, W, 3, padding=2, dilation=2),
            nn.Conv2d(W, W, 3, padding=1)])
        self.ctx = nn.ModuleList([nn.Linear(2 * W, W) for _ in range(3)])
        self.head = nn.Conv2d(W, spec.n_types, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        x = self.stem(obs)
        for conv, ctx in zip(self.convs, self.ctx):
            g = ctx(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
            x = F.relu(x + conv(x) + g[:, :, None, None])
        place = self.head(x).flatten(1)
        gg = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
