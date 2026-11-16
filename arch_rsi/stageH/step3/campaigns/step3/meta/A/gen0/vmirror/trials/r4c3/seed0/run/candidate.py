import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "lr": 0.0015, "n_envs": 128, "rollout_len": 32, "n_minibatches": 32, "gamma": 0.98, "gae_lambda": 0.95, "ent_coef": 0.003}

STEM = 64
COL = 48
WIDTH = 48
HHID = 32
DRED = 8
MF = 32


def mask_views(m, nt, Y, X):
    B = m.shape[0]
    v0 = m.reshape(B, nt, Y, X)
    v1 = m.reshape(B, Y, X, nt).permute(0, 3, 1, 2)
    v2 = m.reshape(B, nt, X, Y).permute(0, 1, 3, 2)
    v3 = m.reshape(B, X, Y, nt).permute(0, 3, 2, 1)
    return torch.cat([v0, v1, v2, v3], dim=1)


def flat_views(ls):
    return [ls[0].flatten(1),
            ls[1].permute(0, 2, 3, 1).flatten(1),
            ls[2].permute(0, 1, 3, 2).flatten(1),
            ls[3].permute(0, 3, 2, 1).flatten(1)]


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + 4 * self.nt, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hid = nn.Conv2d(WIDTH, HHID, 3, padding=1)
        self.outs = nn.ModuleList([nn.Conv2d(HHID, self.nt, 3, padding=1) for _ in range(4)])
        self.red = nn.Conv2d(WIDTH, DRED, 1)
        self.dense = nn.Linear(DRED * self.Y * self.X, self.n_place)
        self.mfeat = nn.Linear(self.n_place, MF)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + MF, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)
        nn.init.normal_(self.dense.weight, std=0.01)
        nn.init.zeros_(self.dense.bias)

    def forward(self, obs, mask):
        m = mask[:, :self.n_place].to(obs.dtype)
        mv = mask_views(m, self.nt, self.Y, self.X)
        s = self.stem(torch.cat([obs, mv], dim=1))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        u = F.relu(self.hid(f))
        ls = [o(u) for o in self.outs]
        views = flat_views(ls)
        place = self.dense(F.relu(self.red(f)).flatten(1))
        for v in views:
            place = place + v
        mf = F.relu(self.mfeat(m))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), mf], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
