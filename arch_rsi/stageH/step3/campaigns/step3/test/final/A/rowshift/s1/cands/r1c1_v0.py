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
PROJ = 12
HID = 12
SCALE = 5.0
PER_TYPE = False


def frame_list(Y, X):
    trs = [False, True] if Y == X else [False]
    return [(tr, fy, fx) for tr in trs for fy in (False, True) for fx in (False, True)]


def apply_frame(x, fr):
    if fr[0]:
        x = x.transpose(2, 3)
    if fr[1]:
        x = torch.flip(x, [2])
    if fr[2]:
        x = torch.flip(x, [3])
    return x


def ortho(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class Trunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        r2 = math.sqrt(2.0)
        self.s1 = ortho(nn.Conv2d(c, STEM, 1), r2)
        self.s2 = ortho(nn.Conv2d(STEM, COL, 1), r2)
        self.c1 = ortho(nn.Conv2d(COL, WIDTH, 3, padding=1), r2)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = ortho(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), r2)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = ortho(nn.Conv2d(c, SKIP, 1), 1.0)
        self.glob = ortho(nn.Linear(2 * WIDTH, GH), r2)
        self.out_ch = WIDTH + SKIP

    def forward(self, obs):
        s = F.relu(self.s2(F.relu(self.s1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)))
        feat = torch.cat([f, self.skip(obs)], 1)
        return feat, g


class FramePoolHead(nn.Module):
    def __init__(self, spec, cin, per_type, scale):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        self.frames = frame_list(spec.Y, spec.X)
        nf = len(self.frames)
        self.nf = nf
        self.scale = scale
        self.proj = ortho(nn.Conv2d(cin, PROJ, 1), 1.0)
        w1 = torch.empty(T, nf, HID, 3 * PROJ)
        w2 = torch.empty(T, nf, HID)
        for t in range(T):
            for f in range(nf):
                nn.init.orthogonal_(w1[t, f], math.sqrt(2.0))
            nn.init.orthogonal_(w2[t], 0.01)
        self.w1 = nn.Parameter(w1)
        self.b1 = nn.Parameter(torch.zeros(T, nf, HID))
        self.w2 = nn.Parameter(w2)
        self.b2 = nn.Parameter(torch.zeros(T, nf))
        self.mix = nn.Parameter(torch.zeros(T if per_type else 1, nf))
        self.tbias = ortho(nn.Linear(GH, T), 0.01)

    def forward(self, feat, g):
        p = self.proj(feat)
        B, P = p.shape[0], p.shape[1]
        pall = torch.cat([apply_frame(p, fr) for fr in self.frames], 1)
        outs = []
        for t, (dy, dx) in enumerate(self.fps):
            if dy * dx == 1:
                st = torch.stack([pall, pall, pall], 1)
            else:
                q = F.pad(pall, (0, dx - 1, 0, dy - 1))
                st = torch.stack([F.max_pool2d(q, (dy, dx), stride=1),
                                  -F.max_pool2d(-q, (dy, dx), stride=1),
                                  F.avg_pool2d(q, (dy, dx), stride=1)], 1)
            st = st.reshape(B, 3, self.nf, P, -1).permute(0, 2, 1, 3, 4).reshape(B, self.nf, 3 * P, -1)
            h = F.relu(torch.einsum('bfcn,fhc->bfhn', st, self.w1[t]) + self.b1[t][None, :, :, None])
            o = torch.einsum('bfhn,fh->bfn', h, self.w2[t]) + self.b2[t][None, :, None]
            outs.append(o)
        L = torch.stack(outs, 1)
        w = F.softmax(self.scale * self.mix, dim=-1)
        L = (L * w[None, :, :, None]).sum(2) + self.tbias(g)[:, :, None]
        return L.flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec)
        self.head = FramePoolHead(spec, self.trunk.out_ch, PER_TYPE, SCALE)
        self.done_head = ortho(nn.Linear(GH, 1), 0.01)
        self.value_head = ortho(nn.Linear(GH, 1), 1.0)

    def forward(self, obs, mask):
        feat, g = self.trunk(obs)
        place = self.head(feat, g)
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
