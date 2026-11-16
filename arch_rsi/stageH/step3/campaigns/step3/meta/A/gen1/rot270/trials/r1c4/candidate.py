import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 32, "ppo_epochs": 2, "n_minibatches": 64, "lr": 0.0015, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.003}

COL1 = 96
COL2 = 64
CTX = 32
PROJ = 24
HID = 48
HID2 = 32
GL = 96
G = 1.4142135623730951


def init_(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        C = spec.in_channels
        self.col1 = init_(nn.Conv2d(C, COL1, 1), G)
        self.col2 = init_(nn.Conv2d(COL1, COL2, 1), G)
        self.ncol = nn.GroupNorm(4, COL2)
        self.t1 = init_(nn.Conv2d(COL2, CTX, 3, padding=1), G)
        self.nt1 = nn.GroupNorm(4, CTX)
        self.t2 = init_(nn.Conv2d(CTX, CTX, 3, padding=1), G)
        self.nt2 = nn.GroupNorm(4, CTX)
        self.proj = init_(nn.Conv2d(COL2 + CTX, PROJ, 1), 1.0)
        self.fconv = nn.ModuleList([init_(nn.Conv2d(PROJ, HID, (dy, dx)), G) for dy, dx in self.fps])
        self.sconv = nn.ModuleList([init_(nn.Conv2d(2 * PROJ, HID, 1), G) for _ in self.fps])
        self.m2 = nn.ModuleList([init_(nn.Conv2d(HID, HID2, 1), G) for _ in self.fps])
        self.outs = nn.ModuleList([init_(nn.Conv2d(HID2, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_(nn.Conv2d(COL2 + CTX, 1, 1), 0.01)
        self.glob = init_(nn.Linear(2 * CTX + 2 * COL2, GL), G)
        self.type_bias = init_(nn.Linear(GL, T), 0.01)
        self.done_head = init_(nn.Linear(GL, 1), 0.01)
        self.value_head = init_(nn.Linear(GL, 1), 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        c = F.relu(self.ncol(self.col2(F.relu(self.col1(obs)))))
        t = F.relu(self.nt1(self.t1(c)))
        t = F.relu(t + self.nt2(self.t2(t)))
        fr = torch.cat([c, t], 1)
        p = self.proj(fr)
        g = F.relu(self.glob(torch.cat([t.mean((2, 3)), t.amax((2, 3)), c.mean((2, 3)), c.amax((2, 3))], 1)))
        tb = self.type_bias(g)
        a = self.anchor(fr)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            if dy * dx == 1:
                mx, mn = p, p
            else:
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            h = F.relu(self.fconv[i](q) + self.sconv[i](torch.cat([mx, mn], 1)))
            h = F.relu(self.m2[i](h))
            res.append(self.outs[i](h) + a + tb[:, i].view(B, 1, 1, 1))
        place = torch.cat(res, 1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
