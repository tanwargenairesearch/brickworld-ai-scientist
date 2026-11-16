import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

HID = 96
K = 8
CTX = 32
CP = 8
MH = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.col1 = nn.Conv2d(spec.in_channels, HID, 1)
        self.col2 = nn.Conv2d(HID, HID, 1)
        self.lay = nn.Conv2d(HID, Z * K, 1)
        self.ctx_in = nn.Conv2d(HID, CTX, 1)
        self.ctx_conv = nn.Conv2d(CTX, CTX, 3, padding=1)
        self.ctx_proj = nn.Conv2d(CTX, CP, 1)
        self.zemb = nn.Parameter(torch.zeros(Z, MH))
        self.mix1 = nn.ModuleList([nn.Linear(3 * K, MH) for _ in self.fps])
        self.cmix = nn.ModuleList([nn.Conv2d(2 * CP, MH, 1) for _ in self.fps])
        self.mix2 = nn.ModuleList([nn.Linear(MH, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * CTX + 2 * Z * K, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        h = F.relu(self.col1(obs))
        h = F.relu(self.col2(h))
        a = self.lay(h)
        c = F.relu(self.ctx_in(h))
        c = F.relu(c + self.ctx_conv(c))
        cp = self.ctx_proj(c)
        res = []
        for (dy, dx), m1, cm, m2 in zip(self.fps, self.mix1, self.cmix, self.mix2):
            if dy * dx == 1:
                mx, mn, av = a, a, a
                cmx, cav = cp, cp
            else:
                q = F.pad(a, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
                qc = F.pad(cp, (0, dx - 1, 0, dy - 1))
                cmx = F.max_pool2d(qc, (dy, dx), stride=1)
                cav = F.avg_pool2d(qc, (dy, dx), stride=1)
            pl = torch.cat([mx.reshape(B, Z, K, Y, X), mn.reshape(B, Z, K, Y, X), av.reshape(B, Z, K, Y, X)], dim=2)
            pl = pl.permute(0, 3, 4, 1, 2)
            cb = cm(torch.cat([cmx, cav], dim=1)).permute(0, 2, 3, 1).unsqueeze(3)
            hz = F.relu(m1(pl) + cb + self.zemb)
            s = m2(hz).squeeze(-1)
            res.append(torch.logsumexp(s, dim=-1).reshape(B, Y * X))
        place = torch.cat(res, dim=1)
        g = self.glob(torch.cat([c.mean((2, 3)), c.amax((2, 3)), a.mean((2, 3)), a.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
