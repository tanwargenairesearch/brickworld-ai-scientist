import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 3, "lr": 0.0007, "ent_coef": 0.01}

STEM = 64
WD = 48
RED = 16
HID = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        self.n_place = int(spec.n_place)
        cin = int(spec.in_channels) + 2
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WD, 1), nn.ReLU())
        self.c1 = nn.Conv2d(WD, WD, 3, padding=1)
        self.c2 = nn.Conv2d(WD, WD, 3, padding=1)
        self.red = nn.Conv2d(WD, RED, 1)
        self.fc = nn.Sequential(nn.Linear(RED * Y * X + 2 * WD, HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU())
        self.pi = nn.Linear(HID, self.n_place + 1)
        self.local = nn.Conv2d(WD, int(spec.n_types), 1)
        self.value_head = nn.Linear(HID, 1)
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.normal_(self.local.weight, std=0.01)
        nn.init.zeros_(self.local.bias)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        s = self.stem(torch.cat([obs, ys, xs], dim=1))
        x = F.relu(s + self.c1(s))
        f = F.relu(x + self.c2(x))
        r = F.relu(self.red(f)).flatten(1)
        g = torch.cat([r, f.mean((2, 3)), f.amax((2, 3))], dim=1)
        h = self.fc(g)
        loc = self.local(f).flatten(1)
        loc = torch.cat([loc, torch.zeros(B, 1, device=obs.device, dtype=obs.dtype)], dim=1)
        logits = self.pi(h) + loc
        return logits, self.value_head(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
