import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 32
SKIP = 16
PROJ = 16
HID = 12
GLOB = 64
SCALE = 5.0
PER_TYPE = False


def frame_tf(x, g):
    if g >= 4:
        x = x.transpose(2, 3)
    m = g % 4
    if m == 1:
        x = x.flip(3)
    elif m == 2:
        x = x.flip(2)
    elif m == 3:
        x = x.flip(2).flip(3)
    return x


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.stem1 = nn.Conv2d(cin, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, COL, 1)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = nn.Conv2d(cin, SKIP, 1)
        for m in [self.stem1, self.stem2, self.c1, self.c2, self.skip]:
            ortho(m, math.sqrt(2))

    def forward(self, obs):
        s = F.relu(self.stem2(F.relu(self.stem1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        k = F.relu(self.skip(obs))
        return torch.cat([x, k], 1), x


class FramePoolHead(nn.Module):
    def __init__(self, cin, fps, n_frames, per_type, scale):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in fps]
        T = len(self.fps)
        G = n_frames
        self.G = G
        self.per_type = per_type
        self.scale = scale
        self.proj = nn.Conv2d(cin, PROJ, 1)
        ortho(self.proj, 1.0)
        c3 = 3 * PROJ
        self.w1 = nn.Parameter(torch.randn(G, T, HID, c3) * math.sqrt(2.0 / c3))
        self.b1 = nn.Parameter(torch.zeros(G, T, HID, 1))
        self.w2 = nn.Parameter(torch.randn(G, T, HID) * (0.01 / math.sqrt(HID)))
        self.b2 = nn.Parameter(torch.zeros(G, T, 1))
        if per_type:
            self.alpha = nn.Parameter(torch.zeros(T, G))
        else:
            self.alpha = nn.Parameter(torch.zeros(G))

    def forward(self, f):
        p = self.proj(f)
        frames = []
        for g in range(self.G):
            q = frame_tf(p, g)
            per_t = []
            for (dy, dx) in self.fps:
                if dy * dx == 1:
                    z = torch.cat([q, q, q], 1)
                else:
                    qq = F.pad(q, (0, dx - 1, 0, dy - 1))
                    mx = F.max_pool2d(qq, (dy, dx), stride=1)
                    mn = -F.max_pool2d(-qq, (dy, dx), stride=1)
                    av = F.avg_pool2d(qq, (dy, dx), stride=1)
                    z = torch.cat([mx, mn, av], 1)
                per_t.append(z.flatten(2))
            frames.append(torch.stack(per_t, 1))
        z = torch.stack(frames, 1)
        h = F.relu(torch.einsum('bgtcn,gthc->bgthn', z, self.w1) + self.b1)
        l = torch.einsum('bgthn,gth->bgtn', h, self.w2) + self.b2
        if self.per_type:
            w = torch.softmax(self.scale * self.alpha, dim=1)
            return torch.einsum('bgtn,tg->btn', l, w)
        w = torch.softmax(self.scale * self.alpha, dim=0)
        return torch.einsum('bgtn,g->btn', l, w)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        G = 8 if Y == X else 4
        T = len(spec.footprints)
        self.trunk = Trunk(spec.in_channels)
        self.head = FramePoolHead(WIDTH + SKIP, spec.footprints, G, PER_TYPE, SCALE)
        self.glob = nn.Linear(2 * WIDTH, GLOB)
        ortho(self.glob, math.sqrt(2))
        self.done_head = nn.Linear(GLOB, 1)
        ortho(self.done_head, 0.01)
        self.value_head = nn.Linear(GLOB, 1)
        ortho(self.value_head, 1.0)
        self.type_bias = nn.Linear(GLOB, T)
        ortho(self.type_bias, 0.01)

    def forward(self, obs, mask):
        f, x = self.trunk(obs)
        g = F.relu(self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)))
        place = self.head(f) + self.type_bias(g).unsqueeze(2)
        place = place.flatten(1)
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
