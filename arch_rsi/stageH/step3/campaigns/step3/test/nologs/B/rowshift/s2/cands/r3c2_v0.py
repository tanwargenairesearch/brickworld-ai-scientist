import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.98, "gae_lambda": 0.95}

W = 16
GHID = 64
NPER = 8


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.GroupNorm(4, c)
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(4, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.relu(self.n1(x)))
        h = self.c2(F.relu(self.n2(h)))
        return x + h


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.T = spec.n_types
        self.n_place = spec.n_place
        cin = spec.in_channels + spec.n_types
        self.shared = nn.Conv2d(cin, W, 3, padding=1)
        self.local = nn.Conv2d(NPER + spec.Z, W, 1)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.block = Block(W)
        self.out_norm = nn.GroupNorm(4, W)
        self.po = nn.Conv2d(W, spec.n_types, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * W, GHID), nn.ReLU())
        self.done_head = nn.Linear(GHID, 1)
        self.vh = nn.Sequential(nn.Linear(GHID, GHID), nn.ReLU())
        self.value_head = nn.Linear(GHID, 1)
        nn.init.orthogonal_(self.po.weight, 0.01)
        nn.init.zeros_(self.po.bias)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        Z = self.Z
        cur = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        ext = obs[:, 3 * Z:4 * Z]
        one = torch.ones_like(cur[:, :1])
        zero = torch.zeros_like(cur[:, :1])
        curb = torch.cat([one, cur[:, :Z - 1]], 1)
        tgtb = torch.cat([one, tgt[:, :Z - 1]], 1)
        missb = torch.cat([zero, miss[:, :Z - 1]], 1)
        cura = torch.cat([cur[:, 1:], zero], 1)
        per = torch.stack([cur, tgt, miss, ext, curb, tgtb, missb, cura], 2)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        per = torch.cat([per, eye], 2).reshape(B * Z, NPER + Z, Y, X)
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.T, Y, X)
        s = self.shared(torch.cat([obs, m], 1))
        s = s.unsqueeze(1).expand(B, Z, W, Y, X).reshape(B * Z, W, Y, X)
        h = F.relu(self.local(per) + s)
        x = self.c1(h)
        x = self.block(x)
        f = F.relu(self.out_norm(x))
        p = self.po(f).reshape(B, Z, self.T, Y, X)
        place = torch.logsumexp(p, dim=1).flatten(1)
        f5 = f.reshape(B, Z, W, Y, X)
        g = self.glob(torch.cat([f5.mean((1, 3, 4)), f5.amax((1, 3, 4))], dim=1))
        done = self.done_head(g)
        value = self.value_head(self.vh(g)).squeeze(1)
        return torch.cat([place, done], dim=1), value


def build_policy(spec):
    return Net(spec)
