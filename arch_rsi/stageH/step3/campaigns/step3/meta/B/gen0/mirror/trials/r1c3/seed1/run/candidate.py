import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.97}

STEM = 64
COL = 48
WIDTH = 40
L = 24
F2 = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = int(spec.Z)
        self.Z = Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.register_buffer("zpos", torch.eye(Z).view(1, Z, Z, 1, 1))
        self.lin = nn.Conv2d(11 + Z, L, 1)
        self.tproj = nn.Conv2d(WIDTH, L, 1)
        self.mid = nn.Conv2d(L, L, 1)
        self.fconv = nn.ModuleList([nn.Conv2d(L, F2, (dy, dx)) for dy, dx in self.fps])
        self.fout = nn.ModuleList([nn.Conv2d(F2, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        T = len(self.fps)
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))

        b = obs[:, 0:Z]
        tg = obs[:, Z:2 * Z]
        ms = obs[:, 2 * Z:3 * Z]
        ex = obs[:, 3 * Z:4 * Z]
        bb = F.pad(b, (0, 0, 0, 0, 1, 0))[:, :Z]
        tb = F.pad(tg, (0, 0, 0, 0, 1, 0))[:, :Z]
        ba = F.pad(b, (0, 0, 0, 0, 0, 1))[:, 1:]
        ta = F.pad(tg, (0, 0, 0, 0, 0, 1))[:, 1:]
        per = torch.stack([b, tg, ms, ex, bb, tb, ba, ta], dim=2)
        gin = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zp = self.zpos.expand(B, Z, Z, Y, X)
        li = torch.cat([per, gin, zp], dim=2).reshape(B * Z, 11 + Z, Y, X)
        tp = self.tproj(f).unsqueeze(1).expand(B, Z, L, Y, X).reshape(B * Z, L, Y, X)
        h = F.relu(self.lin(li) + tp)
        h = F.relu(h + self.mid(h))

        scores = []
        gates = []
        for (dy, dx), cv, out in zip(self.fps, self.fconv, self.fout):
            q = F.pad(h, (0, dx - 1, 0, dy - 1))
            o = out(F.relu(cv(q)))
            scores.append(o[:, 0])
            gates.append(o[:, 1])
        sc = torch.stack(scores, 1).reshape(B, Z, T, Y, X)
        gt = torch.stack(gates, 1).reshape(B, Z, T, Y, X)
        w = torch.softmax(gt, dim=1)
        place = (w * sc).sum(1).flatten(1)

        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
