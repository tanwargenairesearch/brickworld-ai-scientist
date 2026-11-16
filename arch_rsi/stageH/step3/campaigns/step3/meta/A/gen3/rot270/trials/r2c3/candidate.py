import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.003}

WIDTH = 32
SKIP = 16
PROJ = 16
HID = 12
SCALE = 2.0


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


def frame(x, k):
    if k >= 4:
        x = x.transpose(2, 3)
    r = k % 4
    if r == 1:
        x = x.flip(3)
    elif r == 2:
        x = x.flip(2)
    elif r == 3:
        x = x.flip(2).flip(3)
    return x


def frame_list(spec):
    if int(spec.X) == int(spec.Y):
        return list(range(8))
    return [0, 1, 2, 3]


class Trunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        g = math.sqrt(2.0)
        self.stem = nn.Sequential(ortho(nn.Conv2d(c, 64, 1), g), nn.ReLU(),
                                  ortho(nn.Conv2d(64, 48, 1), g), nn.ReLU())
        self.c1 = ortho(nn.Conv2d(48, WIDTH, 3, padding=1), g)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = ortho(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), g)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = ortho(nn.Conv2d(c, SKIP, 1), g)

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        h = torch.cat([f, F.relu(self.skip(obs))], 1)
        return f, h


class FrameHead(nn.Module):
    def __init__(self, cin, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.frames = frame_list(spec)
        nf = len(self.frames)
        T = len(self.fps)
        self.nf = nf
        self.T = T
        self.proj = ortho(nn.Conv2d(cin, PROJ, 1), 1.0)
        self.w1 = nn.Parameter(torch.randn(nf, T, 3 * PROJ, HID) * math.sqrt(2.0 / (3 * PROJ)))
        self.b1 = nn.Parameter(torch.zeros(nf, T, HID))
        self.w2 = nn.Parameter(torch.randn(nf, T, HID) * (0.01 / math.sqrt(HID)))
        self.b2 = nn.Parameter(torch.zeros(nf, T))
        self.mixw = nn.Parameter(torch.zeros(T, nf))

    def forward(self, h):
        p = self.proj(h)
        B, P = p.shape[0], p.shape[1]
        nf = self.nf
        q = torch.cat([frame(p, k) for k in self.frames], 1)
        Yp, Xp = q.shape[2], q.shape[3]
        feats = []
        for dy, dx in self.fps:
            if dy * dx == 1:
                mx, mn, av = q, q, q
            else:
                qq = F.pad(q, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(qq, (dy, dx), stride=1)
                mn = -F.max_pool2d(-qq, (dy, dx), stride=1)
                av = F.avg_pool2d(qq, (dy, dx), stride=1)
            z = torch.cat([mx.reshape(B, nf, P, Yp, Xp), mn.reshape(B, nf, P, Yp, Xp),
                           av.reshape(B, nf, P, Yp, Xp)], 2)
            feats.append(z)
        z = torch.stack(feats, 2)
        hid = torch.einsum('bftcyx,ftch->bfthyx', z, self.w1) + self.b1[None, :, :, :, None, None]
        hid = F.relu(hid)
        L = torch.einsum('bfthyx,fth->bftyx', hid, self.w2) + self.b2[None, :, :, None, None]
        L = L.reshape(B, nf, self.T, Yp * Xp)
        a = torch.softmax(SCALE * self.mixw, dim=1)
        return (L * a.t()[None, :, :, None]).sum(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec)
        self.head = FrameHead(WIDTH + SKIP, spec)
        self.glob = nn.Sequential(ortho(nn.Linear(2 * WIDTH, 64), math.sqrt(2.0)), nn.ReLU())
        self.tbias = ortho(nn.Linear(64, len(spec.footprints)), 0.01)
        self.done_head = ortho(nn.Linear(64, 1), 0.01)
        self.value_head = ortho(nn.Linear(64, 1), 1.0)

    def forward(self, obs, mask):
        f, h = self.trunk(obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        place = self.head(h) + self.tbias(g)[:, :, None]
        logits = torch.cat([place.flatten(1), self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
