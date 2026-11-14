import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 24
HID = 32


class CellHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        ins = []
        for dy, dx in self.fps:
            n = dy * dx
            ins.append(proj * n + (2 * proj if n > 1 else 0))
        self.mix = nn.ModuleList([nn.Conv2d(i, hid, 1) for i in ins])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
        Y, X = p.shape[2], p.shape[3]
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                z = p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                cells = [q[:, :, i:i + Y, j:j + X] for i in range(dy) for j in range(dx)]
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                z = torch.cat(cells + [mx, mn], dim=1)
            res.append(out(F.relu(mix(z))))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.bcast = nn.Linear(64, WIDTH)
        self.head = CellHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob2 = nn.Sequential(nn.Linear(2 * WIDTH + 64, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        f2 = F.relu(f + self.bcast(g)[:, :, None, None])
        place = self.head(f2)
        g2 = self.glob2(torch.cat([f2.mean((2, 3)), f2.amax((2, 3)), g], dim=1))
        logits = torch.cat([place, self.done_head(g2)], dim=1)
        return logits, self.value_head(g2).squeeze(1)


def build_policy(spec):
    return Net(spec)
