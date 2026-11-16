import torch
import torch.nn as nn
import torch.nn.functional as F
import math

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0025}

WIDTH = 40
SKIP = 16
GHID = 64
KC = 12
NH = 4
ASCALE = 5.0


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


class AddressHead(nn.Module):
    def __init__(self, spec, cin):
        super().__init__()
        self.YX = int(spec.Y) * int(spec.X)
        self.nA = int(spec.n_actions)
        self.proj = nn.Conv2d(cin, NH * KC, 1)
        self.addr = nn.Parameter(torch.zeros(NH, self.nA, self.YX))
        self.abias = nn.Parameter(torch.zeros(self.nA, KC))
        self.beta = nn.Parameter(torch.zeros(self.nA, KC))
        self.shared = nn.Linear(KC, 1)

    def forward(self, f):
        B = f.shape[0]
        M = self.proj(f).reshape(B, NH, KC, self.YX)
        A = torch.softmax(ASCALE * self.addr, dim=2)
        N = torch.einsum('hac,bhkc->bak', A, M)
        z = F.relu(N + self.abias)
        return self.shared(z).squeeze(2) + (z * self.beta).sum(2)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        nA = int(spec.n_actions)
        self.trunk = Trunk(spec)
        self.head = AddressHead(spec, self.trunk.out_ch)
        self.dense = nn.Linear(GHID, nA)
        self.value_head = nn.Linear(GHID, 1)
        init_net(self)
        nn.init.orthogonal_(self.head.shared.weight, 0.5)
        nn.init.orthogonal_(self.dense.weight, 0.01)
        nn.init.orthogonal_(self.value_head.weight, 1.0)

    def forward(self, obs, mask):
        f, g = self.trunk(obs)
        logits = self.head(f) + self.dense(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
