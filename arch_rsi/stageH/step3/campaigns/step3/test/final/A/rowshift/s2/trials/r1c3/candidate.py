import torch
import torch.nn as nn
import torch.nn.functional as F
import math

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

WIDTH = 40
SKIP = 16
HID = 24
GHID = 64
SCALE = 5.0


def init_net(m):
    for mod in m.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(mod.weight, math.sqrt(2))
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)


class Trunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = spec.in_channels
        self.stem = nn.Sequential(nn.Conv2d(C, 64, 1), nn.ReLU(), nn.Conv2d(64, 48, 1), nn.ReLU())
        self.c1 = nn.Conv2d(48, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = nn.Conv2d(C, SKIP, 1)
        self.out_ch = WIDTH + SKIP
        self.glob = nn.Sequential(nn.Linear(2 * self.out_ch, GHID), nn.ReLU())

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        f = torch.cat([x, self.skip(obs)], dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        return f, g


class SquareHead(nn.Module):
    def __init__(self, spec, cin):
        super().__init__()
        self.T = int(spec.n_types)
        self.K = max(max(int(a), int(b)) for a, b in spec.footprints)
        self.nfr = 2
        self.sq = nn.ModuleList([nn.Conv2d(cin, HID, self.K) for _ in range(self.nfr)])
        self.read = nn.ModuleList([nn.Conv2d(HID, self.T, 1) for _ in range(self.nfr)])
        self.tmix = nn.Parameter(torch.zeros(self.nfr, self.T, self.T))

    def forward(self, f):
        K = self.K
        res = []
        for i in range(self.nfr):
            ff = f if i == 0 else f.transpose(2, 3)
            q = F.pad(ff, (0, K - 1, 0, K - 1))
            h = F.relu(self.sq[i](q))
            o = self.read[i](h).flatten(2)
            o = o + torch.einsum('st,btn->bsn', self.tmix[i], o)
            res.append(o.flatten(1))
        return torch.stack(res, 1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec)
        self.head = SquareHead(spec, self.trunk.out_ch)
        self.mixw = nn.Parameter(torch.zeros(self.head.nfr))
        self.done_head = nn.Linear(GHID, 1)
        self.value_head = nn.Linear(GHID, 1)
        self.dense = nn.Linear(GHID, int(spec.n_place))
        init_net(self)
        for m in [self.done_head, self.dense] + list(self.head.read):
            nn.init.orthogonal_(m.weight, 0.01)
        nn.init.orthogonal_(self.value_head.weight, 1.0)

    def forward(self, obs, mask):
        f, g = self.trunk(obs)
        L = self.head(f)
        w = torch.softmax(SCALE * self.mixw, 0)
        place = torch.einsum('f,bfn->bn', w, L) + self.dense(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
