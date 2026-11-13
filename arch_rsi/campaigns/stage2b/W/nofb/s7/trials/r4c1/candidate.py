"""r3c3 recipe + three-layer conv body (last dilated, full width) + global context + footprint heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
HID = 32
RECIPE = {"gamma": 0.99, "gae_lambda": 0.95, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2}


def fp_pool(f, dy, dx):
    p = F.pad(f, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(p, (dy, dx), stride=1)
    mn = -F.max_pool2d(-p, (dy, dx), stride=1)
    return mx, mn


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=2, dilation=2), nn.ReLU())
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.ctx = nn.Linear(WIDTH, WIDTH)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * WIDTH, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in self.fps])
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        h = F.relu(f + self.ctx(g)[:, :, None, None])
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            mx, mn = fp_pool(h, dy, dx)
            outs.append(head(torch.cat([h, mx, mn], 1)).flatten(1))
        logits = torch.cat(outs + [self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
