import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ent_coef": 0.03}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 24
HID = 24


def roll_x(t):
    h = t.shape[3] // 2
    return torch.cat([t[..., h:], t[..., :h]], dim=3)


class LandHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.cell = nn.ModuleList([nn.Conv2d(2 * proj, hid, 1) for _ in self.fps])
        self.mix = nn.ModuleList([nn.Conv2d(3 * hid, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
        Y = p.shape[2]
        X = p.shape[3]
        res = []
        for (dy, dx), cell, mix, out in zip(self.fps, self.cell, self.mix, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            if dy * dx == 1:
                a = p
            else:
                a = F.max_pool2d(q, (dy, dx), stride=1)
            hs = []
            for i in range(dy):
                for j in range(dx):
                    c_ij = q[:, :, i:i + Y, j:j + X]
                    hs.append(F.relu(cell(torch.cat([a, c_ij], dim=1))))
            h = torch.stack(hs, dim=0)
            z = torch.cat([h.amin(0), h.mean(0), h.amax(0)], dim=1)
            z = F.relu(mix(z))
            res.append(out(z))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2 * spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = LandHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        inp = torch.cat([obs, roll_x(obs)], dim=1)
        s = self.stem(inp)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
