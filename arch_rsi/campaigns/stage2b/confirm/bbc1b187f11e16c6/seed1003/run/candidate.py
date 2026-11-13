import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
PROJ = 16
HID = 16
RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.T = spec.n_types
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.place_head = nn.Conv2d(WIDTH, self.T, 1)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.head1 = nn.Conv2d(self.T * 2 * PROJ, self.T * HID, 1, groups=self.T)
        self.head2 = nn.Conv2d(self.T * HID, self.T, 1, groups=self.T)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        g = torch.relu(self.proj(f))
        feats = []
        for (dy, dx) in self.footprints:
            gp = F.pad(g, (0, dx - 1, 0, dy - 1))
            feats.append(F.max_pool2d(gp, kernel_size=(dy, dx), stride=1))
            feats.append(F.avg_pool2d(gp, kernel_size=(dy, dx), stride=1))
        h = torch.cat(feats, dim=1)
        place = self.place_head(f) + self.head2(torch.relu(self.head1(h)))
        place = place.flatten(1)
        q = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(q)], dim=1)
        return logits, self.value_head(q).squeeze(1)


def build_policy(spec):
    return Net(spec)
