import math
import torch
import torch.nn as nn

AW = 56
CW = 24
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.actor = nn.Sequential(
            nn.Conv2d(c, AW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(AW, AW, 3, padding=1), nn.ReLU())
        self.place_head = nn.Conv2d(AW, spec.n_types, 1)
        self.aglob = nn.Sequential(nn.Linear(2 * AW, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.critic = nn.Sequential(
            nn.Conv2d(c, CW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(CW, CW, 3, padding=1), nn.ReLU())
        self.cglob = nn.Sequential(nn.Linear(2 * CW, G), nn.ReLU())
        self.value_head = nn.Linear(G, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, math.sqrt(2.0))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.place_head.weight, 0.01)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.orthogonal_(self.value_head.weight, 1.0)

    def forward(self, obs, mask):
        f = self.actor(obs)
        place = self.place_head(f).flatten(1)
        g = self.aglob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        v = self.critic(obs)
        cg = self.cglob(torch.cat([v.mean((2, 3)), v.amax((2, 3))], dim=1))
        return logits, self.value_head(cg).squeeze(1)


def build_policy(spec):
    return Net(spec)
