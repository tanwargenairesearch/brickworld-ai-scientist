import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
DEPTH = 2
PW = 32
HID = 32
CW = 32

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        layers, c_in = [], spec.in_channels
        for _ in range(DEPTH):
            layers += [nn.Conv2d(c_in, WIDTH, 3, padding=1), nn.ReLU()]
            c_in = WIDTH
        self.body = nn.Sequential(*layers)
        self.proj = nn.Conv2d(WIDTH, PW, 1)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.fp_convs = nn.ModuleList([nn.Conv2d(PW, HID, (dy, dx)) for (dy, dx) in self.fps])
        self.pool_convs = nn.ModuleList([nn.Conv2d(2 * PW, HID, 1) for _ in self.fps])
        self.out_convs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.critic = nn.Sequential(
            nn.Conv2d(spec.in_channels, CW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(CW, CW, 3, padding=1), nn.ReLU())
        self.critic_mlp = nn.Sequential(nn.Linear(2 * CW, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, obs, mask):
        f = self.body(obs)
        p = self.proj(f)
        outs = []
        for i, (dy, dx) in enumerate(self.fps):
            pp = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(pp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-pp, (dy, dx), stride=1)
            h = F.relu(self.fp_convs[i](pp) + self.pool_convs[i](torch.cat([mx, mn], dim=1)))
            outs.append(self.out_convs[i](h))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        c = self.critic(obs)
        value = self.critic_mlp(torch.cat([c.mean((2, 3)), c.amax((2, 3))], dim=1)).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
