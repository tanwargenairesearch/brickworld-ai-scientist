import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.0007, "ent_coef": 0.005, "gamma": 0.97}

CW = 64
TW = 32
PROJ = 32
HID = 32
DF = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        c = int(spec.in_channels)
        self.col = nn.Sequential(nn.Conv2d(c, 64, 1), nn.ReLU(), nn.Conv2d(64, CW, 1), nn.ReLU())
        self.c1 = nn.Conv2d(CW, TW, 3, padding=1)
        self.c2 = nn.Conv2d(TW, TW, 3, padding=1)
        self.gctx = nn.Linear(2 * TW, TW)
        self.proj = nn.Conv2d(CW + TW, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.dfeat = nn.Conv2d(CW, DF, 1)
        self.glob = nn.Sequential(nn.Linear(2 * TW + 2 * DF, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        e = self.col(obs)
        x = F.relu(self.c1(e))
        x = F.relu(x + self.c2(x))
        g0 = torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)
        f = F.relu(x + self.gctx(g0)[:, :, None, None])
        p = self.proj(torch.cat([e, f], dim=1))
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
        place = torch.cat(res, dim=1).flatten(1)
        d = self.dfeat(e)
        gd = torch.cat([d.amax((2, 3)), d.amin((2, 3))], dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), gd], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
