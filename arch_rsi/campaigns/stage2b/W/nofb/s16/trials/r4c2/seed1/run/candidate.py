import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
PROJ = 32
HID = 32
GLOB = 64
RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.99, "gae_lambda": 0.95}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.c1 = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=2, dilation=2)
        self.c3 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.film = nn.Linear(GLOB, WIDTH)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.type_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * PROJ, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in self.footprints])
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)

    def forward(self, obs, mask):
        f = F.relu(self.c1(obs))
        f = F.relu(f + self.c2(f))
        f = F.relu(f + self.c3(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = F.relu(f + self.film(g)[:, :, None, None])
        p = self.proj(h)
        outs = []
        for fp, head in zip(self.footprints, self.type_heads):
            dy, dx = fp
            pp = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(pp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-pp, (dy, dx), stride=1)
            av = F.avg_pool2d(pp, (dy, dx), stride=1)
            outs.append(head(torch.cat([mx, mn, av], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
