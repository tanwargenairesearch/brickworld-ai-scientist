import math
import torch
import torch.nn as nn
import torch.nn.functional as F

AW = 56
CW = 24
VH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.actor = nn.Sequential(
            nn.Conv2d(c, AW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(AW, AW, 3, padding=1), nn.ReLU())
        self.place_head = nn.Conv2d(AW, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * AW, AW), nn.ReLU())
        self.done_head = nn.Linear(AW, 1)
        self.critic = nn.Sequential(
            nn.Conv2d(c, CW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(CW, CW, 3, padding=1), nn.ReLU())
        self.vmlp = nn.Sequential(nn.Linear(2 * CW, VH), nn.ReLU())
        self.value_head = nn.Linear(VH, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.place_head.weight, gain=0.01)
        nn.init.orthogonal_(self.done_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, obs, mask):
        f = self.actor(obs)
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        h = self.critic(obs)
        v = self.vmlp(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        return logits, self.value_head(v).squeeze(1)


def build_policy(spec):
    return Net(spec)
