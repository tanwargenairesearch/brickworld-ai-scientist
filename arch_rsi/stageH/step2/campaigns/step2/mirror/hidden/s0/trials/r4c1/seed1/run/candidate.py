import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "lr": 0.002, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 32
LP = 24
LH = 16


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
        self.fproj = nn.Conv2d(WIDTH, LP, 1)
        self.lin = nn.Conv2d(8 + Z, LP, 1)
        self.l2 = nn.Conv2d(LP, LP, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * LP, LH, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(LH, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + LP, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        bp = F.pad(b, (0, 0, 0, 0, 1, 1))
        tp = F.pad(t, (0, 0, 0, 0, 1, 1))
        loc = torch.stack([b, t, m, e, bp[:, 0:Z], tp[:, 0:Z], bp[:, 2:Z + 2], tp[:, 2:Z + 2]], dim=2)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        loc = torch.cat([loc, eye], dim=2).reshape(B * Z, 8 + Z, Y, X)
        fp = self.fproj(f).unsqueeze(1).expand(B, Z, LP, Y, X).reshape(B * Z, LP, Y, X)
        h = F.relu(self.lin(loc) + fp)
        g = F.relu(self.l2(h))
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = g, g, g
            else:
                q = F.pad(g, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            sc = out(F.relu(mix(torch.cat([mx, mn, av], dim=1))))
            sc = sc.reshape(B, Z, Y * X)
            res.append(torch.logsumexp(sc, dim=1))
        place = torch.cat(res, dim=1)
        gl = g.reshape(B, Z, LP, Y, X).amax(dim=(1, 3, 4))
        gv = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), gl], dim=1))
        logits = torch.cat([place, self.done_head(gv)], dim=1)
        return logits, self.value_head(gv).squeeze(1)


def build_policy(spec):
    return Net(spec)
