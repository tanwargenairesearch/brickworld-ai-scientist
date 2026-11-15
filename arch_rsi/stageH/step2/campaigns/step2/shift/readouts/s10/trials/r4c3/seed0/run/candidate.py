import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "ent_coef": 0.02, "n_minibatches": 32}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32


class UnfoldHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix1 = nn.ModuleList([nn.Conv2d(proj * (dy * dx + 1), hid, 1) for dy, dx in self.fps])
        self.mix2 = nn.ModuleList([nn.Conv2d(hid, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
        Y = p.shape[2]
        X = p.shape[3]
        res = []
        for (dy, dx), m1, m2, out in zip(self.fps, self.mix1, self.mix2, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            parts = []
            for i in range(dy):
                for j in range(dx):
                    parts.append(q[:, :, i:i + Y, j:j + X])
            mx = parts[0]
            for t in parts[1:]:
                mx = torch.maximum(mx, t)
            h = F.relu(m1(torch.cat(parts + [mx], dim=1)))
            h = F.relu(m2(h))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.shift = int(spec.X) // 2
        self.stem = nn.Sequential(
            nn.Conv2d(2 * spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = UnfoldHead(WIDTH + COL, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        k = self.shift
        rolled = torch.cat([obs[:, :, :, k:], obs[:, :, :, :k]], dim=3)
        inp = torch.cat([obs, rolled], dim=1)
        s = self.stem(inp)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(torch.cat([f, s], dim=1))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
