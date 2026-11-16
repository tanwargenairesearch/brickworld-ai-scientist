import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
P = 16
MID = 48

RECIPE = {"n_envs": 256, "rollout_len": 16, "n_minibatches": 32, "ppo_epochs": 2}


def fp_pool(p, dy, dx):
    if dy * dx == 1:
        return torch.cat([p, p, p], dim=1)
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return torch.cat([mx, mn, av], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.proj = nn.Conv2d(WIDTH, P, 1)
        self.mix = nn.Conv2d(T * 3 * P + WIDTH, MID, 1)
        names = ["id", "fy", "fx", "pm"]
        if int(spec.Y) == int(spec.X):
            names.append("tr")
        self.names = names
        self.outs = nn.ModuleList([nn.Conv2d(MID, T, 3, padding=1) for _ in names])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        p = self.proj(f)
        pooled = [fp_pool(p, dy, dx) for (dy, dx) in self.fps]
        m = F.relu(self.mix(torch.cat(pooled + [f], dim=1)))
        place = None
        for name, out in zip(self.names, self.outs):
            o = out(m)
            if name == "fy":
                o = o.flip(2)
            elif name == "fx":
                o = o.flip(3)
            elif name == "tr":
                o = o.transpose(2, 3)
            if name == "pm":
                o = o.permute(0, 2, 3, 1).reshape(B, -1)
            else:
                o = o.reshape(B, -1)
            place = o if place is None else place + o
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
