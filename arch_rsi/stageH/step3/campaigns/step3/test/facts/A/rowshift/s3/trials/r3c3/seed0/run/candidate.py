import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'ppo_epochs': 2, 'lr': 0.002, 'ent_coef': 0.02, 'gamma': 0.97, 'gae_lambda': 0.95}

STEM = 64
COL = 48
WIDTH = 40
D = 8
PROJ = 24
HID = 48


def shift_pad(t, dy, dx):
    return F.pad(t, (0, dx - 1, 0, dy - 1))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.gproj = nn.Linear(64, WIDTH)
        self.key = nn.Conv2d(COL, self.Z * D, 1)
        self.val = nn.Conv2d(COL, self.Z * D, 1)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(dy * dx * D + 2 * PROJ, HID, 1) for dy, dx in self.fps])
        self.mix2 = nn.ModuleList([nn.Conv2d(HID, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.orthogonal_(o.weight, 0.01)
            nn.init.zeros_(o.bias)
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        fg = F.relu(f + self.gproj(g)[:, :, None, None])
        k = self.key(s)
        v = self.val(s)
        p = self.proj(fg)
        scale = 1.0 / math.sqrt(Z)
        res = []
        for (dy, dx), mix, mix2, out in zip(self.fps, self.mix, self.mix2, self.outs):
            if dy * dx == 1:
                kk = k
                mx = p
                av = p
            else:
                kk = F.max_pool2d(shift_pad(k, dy, dx), (dy, dx), stride=1)
                q = shift_pad(p, dy, dx)
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            kk = kk.reshape(B, Z, D, Y, X)
            vp = shift_pad(v, dy, dx)
            feats = []
            for i in range(dy):
                for j in range(dx):
                    vv = vp[:, :, i:i + Y, j:j + X].reshape(B, Z, D, Y, X)
                    feats.append((kk * vv).sum(1) * scale)
            h = F.relu(mix(torch.cat(feats + [mx, av], dim=1)))
            h = F.relu(mix2(h)) + h
            res.append(out(h))
        place = torch.cat(res, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
