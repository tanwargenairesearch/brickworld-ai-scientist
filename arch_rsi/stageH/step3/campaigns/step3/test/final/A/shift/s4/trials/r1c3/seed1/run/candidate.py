import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32,
          "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

WIDTH = 32
SKIP = 16
HID = 32
GHID = 64
SCALE = 5.0


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain=gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class Trunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.stem1 = nn.Conv2d(c, 64, 1)
        self.stem2 = nn.Conv2d(64, 48, 1)
        self.c1 = nn.Conv2d(48, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.out_ch = WIDTH + SKIP
        for m in [self.stem1, self.stem2, self.c1, self.c2, self.skip]:
            init_layer(m, math.sqrt(2))

    def forward(self, obs):
        s = F.relu(self.stem2(F.relu(self.stem1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        sk = F.relu(self.skip(obs))
        return torch.cat([x, sk], dim=1), x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = spec.Y
        self.X = spec.X
        self.T = spec.n_types
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.s = max(max(a, b) for a, b in fps)
        self.trunk = Trunk(spec)
        self.sq = nn.ModuleList([nn.Conv2d(self.trunk.out_ch, HID, self.s) for k in range(2)])
        self.outs = nn.ModuleList([nn.Conv2d(HID, self.T, 1) for k in range(2)])
        self.type_mix = nn.Parameter(torch.zeros(self.T, self.T))
        self.glob = nn.Linear(2 * WIDTH, GHID)
        self.done_head = nn.Linear(GHID, 1)
        self.value_head = nn.Linear(GHID, 1)
        self.type_bias = nn.Linear(GHID, self.T)
        self.dense = nn.Linear(GHID, spec.n_place)
        self.frame_logit = nn.Parameter(torch.zeros(2))
        for m in self.sq:
            init_layer(m, math.sqrt(2))
        for m in self.outs:
            init_layer(m, 0.01)
        init_layer(self.glob, math.sqrt(2))
        init_layer(self.done_head, 0.01)
        init_layer(self.type_bias, 0.01)
        init_layer(self.dense, 0.01)
        init_layer(self.value_head, 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        s = self.s
        f, x = self.trunk(obs)
        cands = []
        for k in range(2):
            fk = f if k == 0 else f.transpose(2, 3)
            q = F.pad(fk, (0, s - 1, 0, s - 1))
            o = self.outs[k](F.relu(self.sq[k](q)))
            o = o + torch.einsum('ts,bsyx->btyx', self.type_mix, o)
            cands.append(o.reshape(B, -1))
        L = torch.stack(cands, dim=0)
        w = torch.softmax(SCALE * self.frame_logit, dim=0)
        place = torch.einsum('k,kbn->bn', w, L)
        g = F.relu(self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)))
        place = place + self.dense(g) + self.type_bias(g).repeat_interleave(self.Y * self.X, dim=1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
