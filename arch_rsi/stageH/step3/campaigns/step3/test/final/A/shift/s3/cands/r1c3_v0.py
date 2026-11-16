import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

WIDTH = 36
SKIP = 16
HID = 32
MIX_SCALE = 5.0


def ortho_init(net):
    for mod in net.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(mod.weight, math.sqrt(2.0))
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)


def set_gain(mod, gain):
    nn.init.orthogonal_(mod.weight, gain)
    nn.init.zeros_(mod.bias)


class Trunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.stem = nn.Sequential(nn.Conv2d(c, 64, 1), nn.ReLU(), nn.Conv2d(64, 48, 1), nn.ReLU())
        self.c1 = nn.Conv2d(48, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        k = F.relu(self.skip(obs))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        return torch.cat([f, k], dim=1), g


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(fps)
        self.K = max(max(a, b) for a, b in fps)
        self.nf = 2
        self.trunk = Trunk(spec)
        self.convs = nn.ModuleList([nn.Conv2d(WIDTH + SKIP, HID, self.K) for i in range(self.nf)])
        self.outs = nn.ModuleList([nn.Conv2d(HID, self.T, 1) for i in range(self.nf)])
        self.dense = nn.Linear(64, int(spec.n_place))
        self.type_bias = nn.Linear(64, self.T)
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        ortho_init(self)
        for o in self.outs:
            set_gain(o, 0.01)
        set_gain(self.dense, 0.01)
        set_gain(self.type_bias, 0.01)
        set_gain(self.done_head, 0.01)
        set_gain(self.value_head, 1.0)
        self.tmix = nn.Parameter(torch.zeros(self.nf, self.T, self.T))
        self.mix = nn.Parameter(torch.zeros(self.nf))

    def forward(self, obs, mask):
        h, g = self.trunk(obs)
        B = h.shape[0]
        K = self.K
        tb = self.type_bias(g)[:, :, None, None]
        comps = []
        for k in range(self.nf):
            hk = h if k == 0 else h.transpose(2, 3)
            q = F.pad(hk, (0, K - 1, 0, K - 1))
            a = self.outs[k](F.relu(self.convs[k](q)))
            a = a + torch.einsum('btyx,ts->bsyx', a, self.tmix[k])
            a = a + tb
            comps.append(a.reshape(B, -1))
        st = torch.stack(comps, dim=1)
        w = torch.softmax(MIX_SCALE * self.mix, dim=0)
        place = torch.einsum('bkn,k->bn', st, w) + self.dense(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
