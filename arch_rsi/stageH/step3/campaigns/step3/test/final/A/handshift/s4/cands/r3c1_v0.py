import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 36
SKIP = 16
PROJ = 32
GHID = 64
MIX_SCALE = 5.0
REMAP_BUDGET = 700000


def make_frames(Y, X):
    frames = []
    trs = [False, True] if Y == X else [False]
    for tr in trs:
        for fy in (False, True):
            for fx in (False, True):
                frames.append((tr, fy, fx))
    return frames


def apply_frame(x, fr):
    tr, fy, fx = fr
    if tr:
        x = x.transpose(-1, -2)
    if fy:
        x = x.flip(-2)
    if fx:
        x = x.flip(-1)
    return x


def init_layers(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def init_out(m, gain):
    nn.init.orthogonal_(m.weight, gain=gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class Trunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = int(spec.in_channels)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        yx = self.Y * self.X
        n_in = c * yx
        self.rf = max(1, min(8, REMAP_BUDGET // (n_in * yx)))
        self.remap = nn.Linear(n_in, self.rf * yx)
        self.stem = nn.Sequential(nn.Conv2d(c, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL + self.rf, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GHID), nn.ReLU())
        self.out_ch = WIDTH + SKIP + self.rf

    def forward(self, obs):
        B = obs.shape[0]
        r = F.relu(self.remap(obs.flatten(1))).reshape(B, self.rf, self.Y, self.X)
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(torch.cat([s, r], dim=1))))
        f = F.relu(x + self.n2(self.c2(x)))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = torch.cat([f, F.relu(self.skip(obs)), r], dim=1)
        return h, g


class WindowFrameHead(nn.Module):
    def __init__(self, spec, cin):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        K = max(max(a, b) for a, b in fps)
        self.frames = make_frames(self.Y, self.X)
        self.nf = len(self.frames)
        self.proj = nn.Conv2d(cin, PROJ, 1)
        self.win = nn.Conv2d(PROJ, self.nf * self.T, 2 * K - 1, padding=K - 1)

    def forward(self, h):
        m = self.win(F.relu(self.proj(h)))
        B = m.shape[0]
        m = m.reshape(B, self.nf, self.T, self.Y, self.X)
        outs = [apply_frame(m[:, i], fr) for i, fr in enumerate(self.frames)]
        return torch.stack(outs, 1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.trunk = Trunk(spec)
        self.head = WindowFrameHead(spec, self.trunk.out_ch)
        self.af = nn.Parameter(torch.zeros(self.head.nf))
        self.ao = nn.Parameter(torch.zeros(3))
        self.ad = nn.Parameter(torch.zeros(2))
        self.dense = nn.Linear(GHID, int(spec.n_actions))
        self.done_head = nn.Linear(GHID, 1)
        self.value_head = nn.Linear(GHID, 1)
        init_layers(self)
        init_out(self.head.win, 0.01)
        init_out(self.dense, 0.01)
        init_out(self.done_head, 0.01)
        init_out(self.value_head, 1.0)

    def forward(self, obs, mask):
        h, g = self.trunk(obs)
        M = self.head(h)
        o1 = M.flatten(2)
        o2 = M.permute(0, 1, 3, 4, 2).flatten(2)
        o3 = M.permute(0, 1, 3, 2, 4).flatten(2)
        L = torch.stack([o1, o2, o3], 2)
        wf = torch.softmax(MIX_SCALE * self.af, 0)
        wo = torch.softmax(MIX_SCALE * self.ao, 0)
        wd = torch.softmax(MIX_SCALE * self.ad, 0)
        place = torch.einsum('bfon,f,o->bn', L, wf, wo)
        d = self.done_head(g)
        v_last = torch.cat([place, d], dim=1)
        v_first = torch.cat([d, place], dim=1)
        logits = wd[0] * v_last + wd[1] * v_first + self.dense(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
