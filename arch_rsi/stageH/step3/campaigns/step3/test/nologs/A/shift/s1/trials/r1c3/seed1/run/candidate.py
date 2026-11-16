import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "gamma": 0.99, "gae_lambda": 0.95, "n_minibatches": 32, "lr": 0.001, "ent_coef": 0.01}

COL = 48
CTX = 32
PROJ = 32
HID = 64
HID2 = 32
GH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.col = nn.Sequential(
            nn.Conv2d(spec.in_channels, 64, 1), nn.ReLU(),
            nn.Conv2d(64, COL, 1), nn.ReLU())
        self.ctx = nn.Conv2d(COL, CTX, 3, padding=1)
        self.proj = nn.Conv2d(COL + CTX, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(PROJ * dy * dx, HID, 1) for dy, dx in self.fps])
        self.mix2 = nn.ModuleList([nn.Conv2d(HID, HID2, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID2, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * (COL + CTX), GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        c = self.col(obs)
        h = torch.cat([c, F.relu(self.ctx(c))], dim=1)
        p = F.relu(self.proj(h))
        Y = p.shape[2]
        X = p.shape[3]
        res = []
        for (dy, dx), mix, mix2, out in zip(self.fps, self.mix, self.mix2, self.outs):
            if dy * dx == 1:
                z = p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                z = torch.cat([q[:, :, oy:oy + Y, ox:ox + X] for oy in range(dy) for ox in range(dx)], dim=1)
            u = F.relu(mix2(F.relu(mix(z))))
            res.append(out(u))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
