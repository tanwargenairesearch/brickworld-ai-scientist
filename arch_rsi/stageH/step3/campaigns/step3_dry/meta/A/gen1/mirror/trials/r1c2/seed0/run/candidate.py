import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8}

CW = 48
HID = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cells = spec.Y * spec.X
        na = spec.n_actions
        rc = max(4, min(16, 300000 // (cells * na)))
        self.conv = nn.Sequential(
            nn.Conv2d(spec.in_channels, CW, 1), nn.ReLU(),
            nn.Conv2d(CW, CW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(CW, CW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(CW, rc, 1), nn.ReLU())
        nf = rc * cells + na
        self.fc = nn.Linear(nf, HID)
        self.pi_direct = nn.Linear(nf, na)
        self.pi_h = nn.Linear(HID, na)
        self.v = nn.Linear(HID, 1)
        for m in (self.pi_direct, self.pi_h, self.v):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, obs, mask):
        f = self.conv(obs).flatten(1)
        z = torch.cat([f, mask.to(obs.dtype)], dim=1)
        h = F.relu(self.fc(z))
        logits = self.pi_direct(z) + self.pi_h(h)
        return logits, self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
