import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
GD = 64

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}


class ShiftHead(nn.Module):
    def __init__(self, c, footprints, proj, hid, gdim):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.gproj = nn.Linear(gdim, proj)
        self.mix = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dy * dx * proj, hid, 1), nn.ReLU(),
            nn.Conv2d(hid, hid, 1), nn.ReLU(),
            nn.Conv2d(hid, 1, 1)) for (dy, dx) in self.fps])

    def forward(self, f, g):
        p = self.proj(f) + self.gproj(g)[:, :, None, None]
        Y = p.shape[2]
        X = p.shape[3]
        res = []
        for (dy, dx), mix in zip(self.fps, self.mix):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            parts = []
            for i in range(dy):
                for j in range(dx):
                    parts.append(q[:, :, i:i + Y, j:j + X])
            res.append(mix(torch.cat(parts, dim=1)))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GD), nn.ReLU())
        self.head = ShiftHead(WIDTH + COL, spec.footprints, PROJ, HID, GD)
        self.done_head = nn.Linear(GD, 1)
        self.value_head = nn.Linear(GD, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        place = self.head(torch.cat([f, s], dim=1), g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
