import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "lr": 0.001, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.01}

STEM = 48
WIDTH = 40
PROJ = 32
HID = 32
NBLK = 3


class FpHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.c1 = nn.ModuleList([nn.Conv2d(proj, hid, (dy, dx)) for dy, dx in self.fps])
        self.c2 = nn.ModuleList([nn.Conv2d(hid, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = F.relu(self.proj(f))
        res = []
        for (dy, dx), c1, c2, out in zip(self.fps, self.c1, self.c2, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            h = F.relu(c1(q))
            h = F.relu(c2(h))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(NBLK)])
        self.ctx = nn.ModuleList([nn.Linear(2 * WIDTH, WIDTH) for _ in range(NBLK)])
        self.head = FpHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = self.stem(obs)
        for conv, ctx in zip(self.blocks, self.ctx):
            g = ctx(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
            f = F.relu(f + conv(f) + g[:, :, None, None])
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
