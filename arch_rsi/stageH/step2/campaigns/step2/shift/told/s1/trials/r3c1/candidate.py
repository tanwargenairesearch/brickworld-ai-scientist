import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "ent_coef": 0.02, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 48
COL = 48
LAY = 32
WIDTH = 40
PROJ = 32
HID = 32


class CircRow(nn.Module):
    def __init__(self, cin, cout, X):
        super().__init__()
        self.X = int(X)
        self.conv = nn.Conv2d(cin, cout, (1, self.X))

    def forward(self, x):
        xp = torch.cat([x, x[..., :self.X - 1]], dim=3)
        return self.conv(xp)


class LayerStem(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.circ = CircRow(8, LAY, spec.X)
        self.emb = nn.Parameter(torch.randn(self.Z, LAY) * 0.1)

    def forward(self, obs):
        Z = self.Z
        B, C, Y, X = obs.shape
        build = obs[:, 0:Z]
        below = F.pad(build, (0, 0, 0, 0, 1, 0), value=1.0)[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        glob = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        st = torch.cat([torch.stack([build, below, tgt, miss, extra], dim=2), glob], dim=2)
        h = self.circ(st.reshape(B * Z, 8, Y, X)).reshape(B, Z, LAY, Y, X)
        h = F.relu(h + self.emb.view(1, Z, LAY, 1, 1))
        return torch.cat([h.mean(1), h.amax(1)], dim=1)


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            CircRow(spec.in_channels, STEM, spec.X), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.layers = LayerStem(spec)
        self.fuse = nn.Conv2d(COL + 2 * LAY, COL, 1)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = F.relu(self.fuse(torch.cat([self.stem(obs), self.layers(obs)], dim=1)))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
