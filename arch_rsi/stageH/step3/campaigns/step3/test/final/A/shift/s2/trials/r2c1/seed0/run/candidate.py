import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 32
SKIP = 16
GH = 64
PRE = 56
FSCALE = 3.0
OSCALE = 4.0
SSCALE = 4.0


def frame(m, k):
    if k >= 4:
        m = m.transpose(-1, -2)
    if k % 2 == 1:
        m = m.flip(-1)
    if (k // 2) % 2 == 1:
        m = m.flip(-2)
    return m


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(self.fps)
        self.nf = 8 if spec.Y == spec.X else 4
        K = max(max(a, b) for a, b in self.fps)
        self.stem1 = nn.Conv2d(c, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, COL, 1)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.pre = nn.Conv2d(WIDTH + SKIP, PRE, 1)
        self.win = nn.Conv2d(PRE, self.nf * self.T, 2 * K - 1, padding=K - 1)
        self.glob = nn.Linear(2 * WIDTH, GH)
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)
        self.type_bias = nn.Linear(GH, self.T)
        self.dense = nn.Linear(GH, spec.n_actions)
        for mod in self.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                init_layer(mod, math.sqrt(2))
        init_layer(self.win, 0.01)
        init_layer(self.done_head, 0.01)
        init_layer(self.type_bias, 0.01)
        init_layer(self.dense, 0.01)
        init_layer(self.value_head, 1.0)
        self.fa = nn.Parameter(torch.zeros(self.T, self.nf))
        self.oa = nn.Parameter(torch.zeros(3))
        self.sa = nn.Parameter(torch.zeros(3))

    def layout(self, m, done):
        mk = torch.stack([frame(m[:, k], k) for k in range(self.nf)], 1)
        wf = torch.softmax(FSCALE * self.fa, 1)
        cm = torch.einsum('bktyx,tk->btyx', mk, wf)
        wo = torch.softmax(OSCALE * self.oa, 0)
        place = wo[0] * cm.flatten(1) + wo[1] * cm.permute(0, 2, 3, 1).flatten(1) + wo[2] * cm.permute(0, 2, 1, 3).flatten(1)
        full = torch.cat([place, done], 1)
        ws = torch.softmax(SSCALE * self.sa, 0)
        return ws[0] * full + ws[1] * torch.roll(full, 1, 1) + ws[2] * torch.roll(full, -1, 1)

    def forward(self, obs, mask):
        s = F.relu(self.stem2(F.relu(self.stem1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        feat = torch.cat([f, F.relu(self.skip(obs))], 1)
        g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)))
        h = F.relu(self.pre(feat))
        B, _, Y, X = h.shape
        m = self.win(h).reshape(B, self.nf, self.T, Y, X)
        m = m + self.type_bias(g)[:, None, :, None, None]
        logits = self.layout(m, self.done_head(g)) + self.dense(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
