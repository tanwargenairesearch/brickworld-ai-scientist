import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 64, "rollout_len": 16, "n_minibatches": 8}

LAYER_CH = 24
WIDTH = 40


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        W = WIDTH
        self.layer_conv = nn.Conv2d(9, LAYER_CH, 3, padding=1)
        self.merge = nn.Conv2d(spec.Z * LAYER_CH, W, 1)
        self.convs = nn.ModuleList([nn.Conv2d(W, W, 3, padding=1) for _ in range(2)])
        self.place_head = nn.Conv2d(W, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        build = obs[:, 0:Z]
        target = obs[:, Z:2 * Z]
        missing = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        ground = torch.ones_like(build[:, :1])
        below = torch.cat([ground, build[:, :Z - 1]], dim=1)
        per = torch.stack([build, below, target, missing, extra], dim=2)
        hts = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype)
        zc = zc.view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([per, hts, zc], dim=2).reshape(B * Z, 9, Y, X)
        h = F.relu(self.layer_conv(x)).reshape(B, Z * LAYER_CH, Y, X)
        h = F.relu(self.merge(h))
        for conv in self.convs:
            h = F.relu(h + conv(h))
        place = self.place_head(h).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
