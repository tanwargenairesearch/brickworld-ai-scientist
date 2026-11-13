import torch
import torch.nn as nn
import torch.nn.functional as F

COL = 64
WIDTH = 48
K = 16
HID = 16

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ent_coef": 0.005, "gae_lambda": 0.95}


class FootprintHead(nn.Module):
    def __init__(self, c, footprints):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in footprints]
        n = len(self.fps)
        self.proj = nn.Conv2d(c, K * n, 1)
        self.anchor = nn.Conv2d(c, HID * n, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * K, HID, 1) for _ in self.fps])
        self.out = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = F.relu(self.proj(f))
        a = self.anchor(f)
        outs = []
        for t, (dy, dx) in enumerate(self.fps):
            pt = p[:, t * K:(t + 1) * K]
            if dy > 1 or dx > 1:
                pt = F.pad(pt, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(pt, (dy, dx), stride=1)
                mn = -F.max_pool2d(-pt, (dy, dx), stride=1)
                av = F.avg_pool2d(pt, (dy, dx), stride=1)
            else:
                mx = pt
                mn = pt
                av = pt
            h = F.relu(self.mix[t](torch.cat([mx, mn, av], 1)) + a[:, t * HID:(t + 1) * HID])
            outs.append(self.out[t](h))
        return torch.cat(outs, 1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.col = nn.Sequential(
            nn.Conv2d(spec.in_channels, COL, 1), nn.ReLU(),
            nn.Conv2d(COL, COL, 1), nn.ReLU())
        self.conv1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.skip = nn.Conv2d(COL, WIDTH, 1)
        self.head = FootprintHead(WIDTH, spec.footprints)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        c = self.col(obs)
        h = F.relu(self.conv1(c))
        f = F.relu(self.conv2(h) + self.skip(c))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
