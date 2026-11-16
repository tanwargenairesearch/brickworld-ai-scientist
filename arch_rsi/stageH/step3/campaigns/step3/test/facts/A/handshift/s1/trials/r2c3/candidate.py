import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.003, "n_envs": 1024, "rollout_len": 8, "n_minibatches": 32, "ppo_epochs": 1, "gae_lambda": 0.8, "ent_coef": 0.05}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
M = 6


def pmax(t, dy, dx):
    if dy * dx == 1:
        return t
    q = F.pad(t, (0, dx - 1, 0, dy - 1))
    return F.max_pool2d(q, (dy, dx), stride=1)


def pmin(t, dy, dx):
    return -pmax(-t, dy, dx)


def pavg(t, dy, dx):
    if dy * dx == 1:
        return t
    q = F.pad(t, (0, dx - 1, 0, dy - 1))
    return F.avg_pool2d(q, (dy, dx), stride=1)


class LandHead(nn.Module):
    def __init__(self, c, cin, Z, footprints):
        super().__init__()
        self.Z = Z
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, PROJ, 1)
        self.hconv = nn.Conv2d(cin, Z + 1, 1)
        self.mconv = nn.Conv2d(cin, M * Z, 1)
        nin = 3 * PROJ + 4 * M + 2 * Z
        self.mix = nn.ModuleList([nn.Conv2d(nin, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])

    def forward(self, f, cfeat):
        Z = self.Z
        p = self.proj(f)
        ph = torch.softmax(self.hconv(cfeat), dim=1)
        c = torch.flip(torch.cumsum(torch.flip(ph, [1]), dim=1), [1])
        mf = torch.sigmoid(self.mconv(cfeat))
        B = f.shape[0]
        Y, X = f.shape[-2], f.shape[-1]
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            mx = pmax(p, dy, dx)
            mn = pmin(p, dy, dx)
            av = pavg(p, dy, dx)
            cm = pmax(c, dy, dx)
            cn = pmin(c, dy, dx)
            land = (cm[:, :Z] - cm[:, 1:]).unsqueeze(1)
            above = cm[:, 1:].unsqueeze(1)
            mmx = pmax(mf, dy, dx).reshape(B, M, Z, Y, X)
            mmn = pmin(mf, dy, dx).reshape(B, M, Z, Y, X)
            s1 = (land * mmn).sum(2)
            s2 = (land * mmx).sum(2)
            s3 = (above * mmx).sum(2)
            s4 = (above * mmn).sum(2)
            h = F.relu(mix(torch.cat([mx, mn, av, s1, s2, s3, s4, cm[:, 1:], cn[:, 1:]], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1), mf


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = int(spec.Z)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = LandHead(WIDTH, spec.in_channels + COL, Z, spec.footprints)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + M * Z, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place, mf = self.head(f, torch.cat([obs, s], dim=1))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), mf.mean((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
