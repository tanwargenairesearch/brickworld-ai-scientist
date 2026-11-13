import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
GW = 64
PROJ = 16
HID = 16
RECIPE = {"n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "gamma": 0.98, "gae_lambda": 0.95}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.T = spec.n_types
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.conv3 = nn.Conv2d(WIDTH, WIDTH, 3, padding=2, dilation=2)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GW), nn.ReLU())
        self.film = nn.Linear(GW, WIDTH)
        self.place_head = nn.Conv2d(WIDTH, self.T, 1)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.head1 = nn.Conv2d(self.T * 2 * PROJ, self.T * HID, 1, groups=self.T)
        self.head2 = nn.Conv2d(self.T * HID, self.T, 1, groups=self.T)
        self.done_head = nn.Linear(GW, 1)
        self.value_head = nn.Linear(GW, 1)

    def forward(self, obs, mask):
        f = torch.relu(self.stem(obs))
        f = f + torch.relu(self.conv2(f))
        f = f + torch.relu(self.conv3(f))
        q = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = torch.relu(f + self.film(q)[:, :, None, None])
        g = torch.relu(self.proj(h))
        feats = []
        for (dy, dx) in self.footprints:
            gp = F.pad(g, (0, dx - 1, 0, dy - 1))
            feats.append(F.max_pool2d(gp, kernel_size=(dy, dx), stride=1))
            feats.append(F.avg_pool2d(gp, kernel_size=(dy, dx), stride=1))
        p = torch.cat(feats, dim=1)
        place = self.place_head(h) + self.head2(torch.relu(self.head1(p)))
        place = place.flatten(1)
        logits = torch.cat([place, self.done_head(q)], dim=1)
        return logits, self.value_head(q).squeeze(1)


def build_policy(spec):
    return Net(spec)
