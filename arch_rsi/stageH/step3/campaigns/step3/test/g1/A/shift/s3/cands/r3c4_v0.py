import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
HID = 32
DCH = 4
GW = 64
DONE_SCALE = 0.1

RECIPE = {"lr": 0.001, "ppo_epochs": 2, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = spec.n_types
        self.n_place = spec.n_place
        C = spec.in_channels + spec.n_types + 2
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            init_layer(nn.Conv2d(C, STEM, 1)), nn.ReLU(),
            init_layer(nn.Conv2d(STEM, COL, 1)), nn.ReLU())
        self.c1 = init_layer(nn.Conv2d(COL, WIDTH, 3, padding=1))
        self.gn1 = nn.GroupNorm(8, WIDTH)
        self.c2 = init_layer(nn.Conv2d(WIDTH, WIDTH, 3, padding=1))
        self.gn2 = nn.GroupNorm(8, WIDTH)
        self.exact = nn.ModuleList([init_layer(nn.Conv2d(WIDTH, HID, (dy, dx))) for dy, dx in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.feat = nn.ModuleList([init_layer(nn.Conv2d(HID, 1, 1), 1.0) for _ in self.fps])
        self.mixer = nn.Linear(spec.n_place, spec.n_place, bias=False)
        nn.init.zeros_(self.mixer.weight)
        self.dproj = init_layer(nn.Conv2d(WIDTH, DCH, 1))
        self.dense = init_layer(nn.Linear(DCH * spec.Y * spec.X, spec.n_place), 0.01)
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * WIDTH, GW)), nn.ReLU())
        self.type_term = init_layer(nn.Linear(GW, spec.n_types), 0.01)
        self.done = init_layer(nn.Linear(GW, 1), 0.01)
        self.value = init_layer(nn.Linear(GW, 1), 1.0)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.T, Y, X)
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        s = self.stem(torch.cat([obs, m, yy, xx], 1))
        x = F.relu(self.gn1(self.c1(s)))
        f = F.relu(x + self.gn2(self.c2(x)))
        res = []
        vs = []
        for (dy, dx), ex, out, ft in zip(self.fps, self.exact, self.outs, self.feat):
            h = F.relu(ex(F.pad(f, (0, dx - 1, 0, dy - 1))))
            res.append(out(h))
            vs.append(ft(h))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        loc = torch.cat(res, 1) + self.type_term(g)[:, :, None, None]
        v = torch.cat(vs, 1).flatten(1)
        d = F.relu(self.dproj(f)).flatten(1)
        place = loc.flatten(1) + self.mixer(v) + self.dense(d)
        done = DONE_SCALE * self.done(g)
        logits = torch.cat([place, done], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
