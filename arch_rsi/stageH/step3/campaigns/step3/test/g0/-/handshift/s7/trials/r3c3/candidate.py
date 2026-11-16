import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "lr": 0.001, "ent_coef": 0.02, "gamma": 0.97, "gae_lambda": 0.9}

CF = 48
HK = 16
H1 = 48
H2 = 24
AG = 32
TW = 32
DONE_BIAS = -3.0


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = int(spec.in_channels)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        n = len(self.fps)
        self.stem = nn.Sequential(
            nn.Conv2d(C, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 1), nn.ReLU(),
            nn.Conv2d(64, CF, 1), nn.ReLU())
        self.hcode = nn.Conv2d(CF, HK, 1)
        self.cellp = nn.Conv2d(CF, H1, 1)
        self.c1 = nn.Conv2d(CF, TW, 3, padding=1)
        self.c2 = nn.Conv2d(TW, TW, 3, padding=1)
        self.ctxp = nn.Conv2d(TW, H1, 1)
        self.fpp = nn.ModuleList([nn.Conv2d(HK, H1, 1) for _ in range(n)])
        self.w2 = nn.ModuleList([nn.Conv2d(H1, H2, 1) for _ in range(n)])
        self.agg = nn.ModuleList([nn.Conv2d(3 * H2, AG, 1) for _ in range(n)])
        self.outs = nn.ModuleList([nn.Conv2d(AG, 1, 1) for _ in range(n)])
        self.glob = nn.Sequential(nn.Linear(2 * (CF + TW), 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        nn.init.constant_(self.done_head.bias, DONE_BIAS)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        hc = self.hcode(s)
        cp = self.cellp(s)
        ctx = self.ctxp(f)
        res = []
        for k, (dy, dx) in enumerate(self.fps):
            if dy * dx == 1:
                pm = hc
                cpp = cp
            else:
                pm = F.max_pool2d(F.pad(hc, (0, dx - 1, 0, dy - 1)), (dy, dx), stride=1)
                cpp = F.pad(cp, (0, dx - 1, 0, dy - 1))
            a = self.fpp[k](pm) + ctx
            vs = []
            for i in range(dy):
                for j in range(dx):
                    u = F.relu(cpp[:, :, i:i + Y, j:j + X] + a)
                    vs.append(F.relu(self.w2[k](u)))
            v = torch.stack(vs, 0)
            ag = torch.cat([v.amax(0), v.amin(0), v.mean(0)], dim=1)
            res.append(self.outs[k](F.relu(self.agg[k](ag))))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([s.mean((2, 3)), s.amax((2, 3)), f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
