import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.97}

HID = 64
K = 12
SPH = 40
GH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.T = int(spec.n_types)
        Z = self.Z
        W = 2 * Z - 1
        cin = 5 * W + 4
        self.feat = nn.Conv2d(cin, HID, 1)
        self.q = nn.Conv2d(HID, K, 1)
        self.sp = nn.Conv2d(K, SPH, 3, padding=1)
        self.out = nn.Conv2d(SPH, self.T, 1)
        self.glob = nn.Sequential(nn.Linear(2 * HID, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def columns(self, obs):
        B, C, Y, X = obs.shape
        Z = self.Z
        W = 2 * Z - 1
        st = obs[:, 0:4 * Z].reshape(B, 4, Z, Y, X)
        ones = torch.ones(B, 1, Z, Y, X, dtype=obs.dtype, device=obs.device)
        st = torch.cat([st, ones], dim=1)
        p = F.pad(st, (0, 0, 0, 0, Z - 1, Z - 1))
        extra = obs[:, 4 * Z:4 * Z + 3]
        wins = []
        for L in range(Z):
            w = p[:, :, L:L + W].reshape(B, 5 * W, Y, X)
            lvl = torch.full((B, 1, Y, X), float(L) / Z, dtype=obs.dtype, device=obs.device)
            wins.append(torch.cat([w, extra, lvl], dim=1))
        u = torch.stack(wins, dim=1)
        return u.reshape(B * Z, 5 * W + 4, Y, X)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        Z = self.Z
        u = self.columns(obs)
        h = F.relu(self.feat(u))
        q = self.q(h)
        s = self.out(F.relu(self.sp(q)))
        s = s.reshape(B, Z, self.T, Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        hh = h.reshape(B, Z, HID, Y, X)
        g = self.glob(torch.cat([hh.mean((1, 3, 4)), hh.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
