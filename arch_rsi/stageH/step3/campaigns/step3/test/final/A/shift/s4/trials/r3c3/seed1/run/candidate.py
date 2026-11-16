import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32,
          "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.01, "lr": 0.001}

WIDTH = 32
SKIP = 16
HID = 40
GHID = 64
CHID = 128
SCALE = 5.0


def frame(f, k):
    if k >= 4:
        f = f.transpose(2, 3)
    if k % 2 == 1:
        f = torch.flip(f, dims=[2])
    if (k // 2) % 2 == 1:
        f = torch.flip(f, dims=[3])
    return f


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain=gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class Trunk(nn.Module):
    def __init__(self, cin, craw, stem):
        super().__init__()
        self.stem1 = nn.Conv2d(cin, stem, 1)
        self.stem2 = nn.Conv2d(stem, 48, 1)
        self.c1 = nn.Conv2d(48, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = nn.Conv2d(craw, SKIP, 1)
        self.out_ch = WIDTH + SKIP
        for m in [self.stem1, self.stem2, self.c1, self.c2, self.skip]:
            init_layer(m, math.sqrt(2))

    def forward(self, o, obs):
        s = F.relu(self.stem2(F.relu(self.stem1(o))))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        sk = F.relu(self.skip(obs))
        return torch.cat([x, sk], dim=1), x, sk


class BroadHead(nn.Module):
    def __init__(self, spec, cin, hid):
        super().__init__()
        self.T = spec.n_types
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        K = max(max(a, b) for a, b in fps)
        self.nf = 8 if spec.Y == spec.X else 4
        self.hid = nn.Conv2d(cin, hid, 2 * K - 1, padding=K - 1)
        self.out = nn.Conv2d(hid, self.nf * self.T, 1)
        self.frame_logit = nn.Parameter(torch.zeros(self.nf))
        self.order_logit = nn.Parameter(torch.zeros(3))
        self.done_logit = nn.Parameter(torch.zeros(2))
        init_layer(self.hid, math.sqrt(2))
        init_layer(self.out, 0.01)

    def forward(self, f, done):
        B = f.shape[0]
        T = self.T
        M = self.out(F.relu(self.hid(f)))
        cands = []
        for k in range(self.nf):
            Lk = frame(M[:, k * T:(k + 1) * T], k)
            orders = [Lk.reshape(B, -1),
                      Lk.permute(0, 2, 1, 3).reshape(B, -1),
                      Lk.permute(0, 2, 3, 1).reshape(B, -1)]
            for o in orders:
                cands.append(torch.cat([o, done], dim=1))
                cands.append(torch.cat([done, o], dim=1))
        L = torch.stack(cands, dim=0)
        wf = torch.softmax(SCALE * self.frame_logit, dim=0)
        wo = torch.softmax(SCALE * self.order_logit, dim=0)
        wd = torch.softmax(SCALE * self.done_logit, dim=0)
        w = (wf[:, None, None] * wo[None, :, None] * wd[None, None, :]).reshape(-1)
        return torch.einsum('k,kbn->bn', w, L)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec.in_channels, spec.in_channels, 64)
        self.head = BroadHead(spec, self.trunk.out_ch, HID)
        self.glob = nn.Linear(2 * WIDTH, GHID)
        self.done_head = nn.Linear(GHID, 1)
        self.dense = nn.Linear(GHID, spec.n_actions)
        self.c_1 = nn.Linear(2 * WIDTH + 2 * SKIP, CHID)
        self.c_2 = nn.Linear(CHID, CHID)
        self.value_head = nn.Linear(CHID, 1)
        init_layer(self.glob, math.sqrt(2))
        init_layer(self.done_head, 0.01)
        init_layer(self.dense, 0.01)
        init_layer(self.c_1, math.sqrt(2))
        init_layer(self.c_2, math.sqrt(2))
        init_layer(self.value_head, 1.0)

    def forward(self, obs, mask):
        f, x, sk = self.trunk(obs, obs)
        pooled = torch.cat([x.mean((2, 3)), x.amax((2, 3)), sk.mean((2, 3)), sk.amax((2, 3))], dim=1)
        g = F.relu(self.glob(pooled[:, :2 * WIDTH]))
        logits = self.head(f, self.done_head(g)) + self.dense(g)
        c = F.relu(self.c_1(pooled.detach()))
        c = F.relu(self.c_2(c))
        return logits, self.value_head(c).squeeze(1)


def build_policy(spec):
    return Net(spec)
