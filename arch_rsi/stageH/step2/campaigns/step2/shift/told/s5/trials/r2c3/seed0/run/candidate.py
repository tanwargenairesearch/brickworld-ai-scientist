import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "n_minibatches": 16, "ppo_epochs": 3, "gamma": 0.97}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32


def xshifts(t):
    X = t.shape[-1]
    outs = [t]
    for k in range(1, X):
        outs.append(torch.cat([t[..., k:], t[..., :k]], dim=-1))
    return torch.cat(outs, dim=1)


def aligned_input(obs, Z):
    build = torch.cat([obs[:, :Z], obs[:, 4 * Z:4 * Z + 1], obs[:, 4 * Z + 2:4 * Z + 3]], dim=1)
    targ = torch.cat([obs[:, Z:4 * Z], obs[:, 4 * Z + 1:4 * Z + 2]], dim=1)
    return torch.cat([build, xshifts(targ)], dim=1)


class CellHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        mlps = []
        for dy, dx in self.fps:
            n = dy * dx
            in_ch = n * proj + (proj if n > 1 else 0)
            mlps.append(nn.Sequential(
                nn.Conv2d(in_ch, hid, 1), nn.ReLU(),
                nn.Conv2d(hid, hid, 1), nn.ReLU(),
                nn.Conv2d(hid, 1, 1)))
        self.mlps = nn.ModuleList(mlps)

    def forward(self, f):
        p = self.proj(f)
        Y, X = p.shape[-2], p.shape[-1]
        res = []
        for (dy, dx), mlp in zip(self.fps, self.mlps):
            if dy * dx == 1:
                z = p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                parts = []
                for i in range(dy):
                    for j in range(dx):
                        parts.append(q[:, :, i:i + Y, j:j + X])
                parts.append(F.max_pool2d(q, (dy, dx), stride=1))
                z = torch.cat(parts, dim=1)
            res.append(mlp(z))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        in_ch = spec.Z + 2 + spec.X * (3 * spec.Z + 1)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = CellHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(aligned_input(obs, self.Z))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
