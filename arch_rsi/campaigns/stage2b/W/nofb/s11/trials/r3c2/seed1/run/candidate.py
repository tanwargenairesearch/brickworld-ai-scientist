import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
HID = 32

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "gamma": 0.99, "gae_lambda": 0.95}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(a), int(b)) for (a, b) in spec.footprints]
        self.conv1 = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.g2c = nn.Linear(WIDTH, WIDTH)
        self.proj = nn.ModuleList([nn.Conv2d(WIDTH, HID, 1) for _ in self.footprints])
        self.mix = nn.ModuleList([nn.Conv2d(2 * HID, HID, 1) for _ in self.footprints])
        self.out = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.footprints])
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = F.relu(self.conv1(obs))
        f = F.relu(self.conv2(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        fc = f + self.g2c(g)[:, :, None, None]
        outs = []
        for i, (dy, dx) in enumerate(self.footprints):
            h = self.proj[i](fc)
            hp = F.pad(h, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(hp, kernel_size=(dy, dx), stride=1)
            mn = -F.max_pool2d(-hp, kernel_size=(dy, dx), stride=1)
            z = F.relu(self.mix[i](torch.cat([mx, mn], dim=1)))
            outs.append(self.out[i](z).flatten(1))
        place = torch.cat(outs, dim=1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
