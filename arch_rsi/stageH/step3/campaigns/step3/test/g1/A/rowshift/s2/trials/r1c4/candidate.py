import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 64, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 4, "ent_coef": 0.003, "gamma": 0.97, "gae_lambda": 0.9, "clip_eps": 0.2}

H1 = 96
H2 = 64
RAW = 16
HID = 40
DF = 16
GH = 128


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class ChanLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c_in = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        g = math.sqrt(2.0)
        self.s1 = init_layer(nn.Conv2d(c_in, H1, 1), g)
        self.n1 = ChanLN(H1)
        self.s2 = init_layer(nn.Conv2d(H1, H2, 1), g)
        self.n2 = ChanLN(H2)
        self.s3 = init_layer(nn.Conv2d(H2, H2, 1), g)
        self.n3 = ChanLN(H2)
        self.raw = init_layer(nn.Conv2d(c_in, RAW, 1), 1.0)
        fd = H2 + RAW
        self.fconv = nn.ModuleList([init_layer(nn.Conv2d(fd, HID, (dy, dx)), g) for dy, dx in self.fps])
        self.mix = nn.ModuleList([init_layer(nn.Conv2d(HID, HID, 1), g) for _ in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_layer(nn.Conv2d(fd, 1, 1), 0.01)
        self.tbias = nn.Parameter(torch.zeros(len(self.fps)))
        self.dfeat = init_layer(nn.Conv2d(fd, DF, 1), g)
        self.dmlp = nn.Sequential(init_layer(nn.Linear(2 * DF, 32), g), nn.ReLU(),
                                  init_layer(nn.Linear(32, 1), 0.01))
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * fd, GH), g), nn.ReLU())
        self.value = init_layer(nn.Linear(GH, 1), 1.0)

    def forward(self, obs, mask):
        s = F.relu(self.n1(self.s1(obs)))
        s = F.relu(self.n2(self.s2(s)))
        s = s + F.relu(self.n3(self.s3(s)))
        f = torch.cat([s, self.raw(obs)], dim=1)
        a = self.anchor(f)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            h = F.relu(self.fconv[i](q))
            h = F.relu(self.mix[i](h))
            res.append(self.outs[i](h) + a + self.tbias[i])
        place = torch.cat(res, dim=1).flatten(1)
        d = F.relu(self.dfeat(f))
        done = self.dmlp(torch.cat([d.amax((2, 3)), d.mean((2, 3))], dim=1))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, done], dim=1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
