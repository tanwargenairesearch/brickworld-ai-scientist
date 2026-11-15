import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 32, "n_minibatches": 32, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.03}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 24
HID = 32


def pool3(p, dy, dx):
    if dy * dx == 1:
        return [p, p, p]
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return [mx, mn, av]


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        na = int(spec.n_actions)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(6 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 3, padding=1) for _ in self.fps])
        self.mfeat = nn.Sequential(nn.Linear(na, 64), nn.ReLU())
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + 64, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        self.remap = nn.Linear(na, na)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            feats = pool3(p, dy, dx) + pool3(p, dx, dy)
            h = F.relu(mix(torch.cat(feats, dim=1)))
            res.append(out(h))
        place = torch.cat(res, dim=1).flatten(1)
        mv = self.mfeat(mask.to(obs.dtype))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), mv], dim=1))
        z = torch.cat([place, self.done_head(g)], dim=1)
        logits = z + self.remap(z)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
