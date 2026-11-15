import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 512, "rollout_len": 16, "n_minibatches": 64, "lr": 0.002, "ent_coef": 0.005}

STEM = 64
COL = 48
WIDTH = 40
HID = 64


def make_frames(square):
    fr = []
    for tr in ([False, True] if square else [False]):
        for fy in (False, True):
            for fx in (False, True):
                fr.append((fy, fx, tr))
    return fr


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.frames = make_frames(int(spec.X) == int(spec.Y))
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HID, 3, padding=1)
        self.h2 = nn.Conv2d(HID, HID, 1)
        self.out = nn.Conv2d(HID, len(self.frames) * 2 * self.T, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        h = F.relu(self.h1(f))
        h = F.relu(self.h2(h))
        o = self.out(h)
        T = self.T
        place = None
        for i, (fy, fx, tr) in enumerate(self.frames):
            for j in range(2):
                k = (2 * i + j) * T
                m = o[:, k:k + T]
                if fy:
                    m = m.flip(2)
                if fx:
                    m = m.flip(3)
                if tr:
                    m = m.transpose(2, 3)
                if j == 0:
                    v = m.flatten(1)
                else:
                    v = m.permute(0, 2, 3, 1).flatten(1)
                place = v if place is None else place + v
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
