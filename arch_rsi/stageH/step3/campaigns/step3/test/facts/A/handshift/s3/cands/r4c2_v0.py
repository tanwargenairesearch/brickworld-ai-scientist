import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 1024, "rollout_len": 8, "n_minibatches": 32, "ppo_epochs": 1, "gamma": 0.95, "gae_lambda": 0.85, "ent_coef": 0.005}

STEM = 48
FEAT = 32
HID = 32
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(C, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, FEAT, 3, padding=1), nn.ReLU())
        self.heads = nn.ModuleList([nn.Conv2d(C + FEAT, HID, (dy + 2, dx + 2)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * FEAT, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        u = torch.cat([obs, s], dim=1)
        res = []
        for (dy, dx), conv, out in zip(self.fps, self.heads, self.outs):
            q = F.pad(u, (1, dx, 1, dy))
            h = F.relu(conv(q))
            res.append(out(h))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([s.mean((2, 3)), s.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
