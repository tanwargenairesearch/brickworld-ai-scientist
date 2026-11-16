import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0025}

STEM = 64
COL = 48
WIDTH = 32
SKIP = 16
GHID = 64
HH = 32
NF = 8
NO = 2
MIX_SCALE = 5.0
RG = 1.4142135623730951


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.s1 = init_layer(nn.Conv2d(cin, STEM, 1), RG)
        self.s2 = init_layer(nn.Conv2d(STEM, COL, 1), RG)
        self.c1 = init_layer(nn.Conv2d(COL, WIDTH, 3, padding=1), RG)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = init_layer(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), RG)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = init_layer(nn.Conv2d(cin, SKIP, 1), RG)
        self.out_ch = WIDTH + SKIP

    def forward(self, obs):
        s = F.relu(self.s2(F.relu(self.s1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        k = F.relu(self.skip(obs))
        return torch.cat([x, k], 1)


def to_frame(m, g):
    if g >= 4:
        m = m.transpose(3, 4)
    k = g % 4
    if k == 1:
        m = m.flip([4])
    elif k == 2:
        m = m.flip([3])
    elif k == 3:
        m = m.flip([3, 4])
    return m


class WindowFrames(nn.Module):
    def __init__(self, c, T, K):
        super().__init__()
        self.T = T
        self.hid = init_layer(nn.Conv2d(c, HH, 1), RG)
        self.win = init_layer(nn.Conv2d(HH, NF * NO * T, 2 * K - 1, padding=K - 1), 0.01)

    def forward(self, f, tb):
        B, _, Y, X = f.shape
        m = self.win(F.relu(self.hid(f))).reshape(B, NF, NO, self.T, Y, X)
        m = m + tb[:, None, None, :, None, None]
        outs = []
        for g in range(NF):
            mg = to_frame(m[:, g], g)
            outs.append(mg[:, 0].reshape(B, -1))
            outs.append(mg[:, 1].permute(0, 2, 3, 1).reshape(B, -1))
        return torch.stack(outs, 0)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        K = max(max(int(a), int(b)) for a, b in spec.footprints)
        self.trunk = Trunk(spec.in_channels)
        c = self.trunk.out_ch
        self.head = WindowFrames(c, self.T, K)
        self.alpha = nn.Parameter(torch.zeros(NF * NO))
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * c, GHID), RG), nn.ReLU())
        self.done_head = init_layer(nn.Linear(GHID, 1), 0.01)
        self.value_head = init_layer(nn.Linear(GHID, 1), 1.0)
        self.type_bias = init_layer(nn.Linear(GHID, self.T), 0.01)
        self.dense = init_layer(nn.Linear(GHID, int(spec.n_place)), 0.01)

    def forward(self, obs, mask):
        f = self.trunk(obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        L = self.head(f, self.type_bias(g))
        w = torch.softmax(MIX_SCALE * self.alpha, 0)
        place = torch.einsum('g,gbn->bn', w, L) + self.dense(g)
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
