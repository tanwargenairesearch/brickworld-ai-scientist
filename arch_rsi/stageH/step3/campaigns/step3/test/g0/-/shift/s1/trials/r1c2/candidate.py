import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_minibatches": 64, "gamma": 0.97, "ent_coef": 0.005}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 16
LD = 6
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.layer_in = nn.Linear(4 + Z, LD)
        self.layer_ctx = nn.Conv2d(WIDTH, LD * Z, 1)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        cin = 2 * (LD * Z + PROJ)
        self.mix = nn.ModuleList([nn.Conv2d(cin, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + LD * Z, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        lay = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 2, 3, 4, 1)
        pos = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, 1, 1, Z).expand(B, Z, Y, X, Z)
        u = self.layer_in(torch.cat([lay, pos], dim=-1))
        u = u.permute(0, 1, 4, 2, 3).reshape(B, Z * LD, Y, X)
        u = F.relu(u + self.layer_ctx(f))
        p = self.proj(f)
        h = torch.cat([u, p], dim=1)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn = h, h
            else:
                q = F.pad(h, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            z = F.relu(mix(torch.cat([mx, mn], dim=1)))
            res.append(out(z))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), u.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
