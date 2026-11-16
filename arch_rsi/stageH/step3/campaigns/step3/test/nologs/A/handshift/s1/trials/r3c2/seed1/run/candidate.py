import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 32, "n_minibatches": 32, "gamma": 0.99, "gae_lambda": 0.95, "lr": 0.001, "ent_coef": 0.01}

STEM = 64
FEAT = 48
PROJ = 32
HID = 48
GL = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, FEAT, 1), nn.ReLU())
        self.c3 = nn.Conv2d(FEAT, FEAT, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * FEAT, GL), nn.ReLU())
        self.proj = nn.Conv2d(FEAT, PROJ, 1)
        self.mix1 = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.film = nn.ModuleList([nn.Linear(GL, HID) for _ in self.fps])
        self.mix2 = nn.ModuleList([nn.Conv2d(HID, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)
        self.done_head = nn.Linear(GL, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(GL, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        f = F.relu(s + self.c3(s))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        p = self.proj(f)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = self.mix1[i](torch.cat([mx, mn, av], dim=1)) + self.film[i](g)[:, :, None, None]
            h = F.relu(h)
            h = F.relu(self.mix2[i](h))
            res.append(self.outs[i](h))
        place = torch.cat(res, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
