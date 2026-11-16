import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 32
SKIP = 16
HID = 24
GH = 64
SCALE = 5.0
HEADS = 4
DH = 8
EMB = 32
QSCALE = 3.0


def frame_list(Y, X):
    trs = [False, True] if Y == X else [False]
    out = []
    for tr in trs:
        for fy in (False, True):
            for fx in (False, True):
                out.append((tr, fy, fx))
    return out


def to_frame(t, fr):
    tr, fy, fx = fr
    if tr:
        t = t.transpose(-1, -2)
    dims = []
    if fy:
        dims.append(-2)
    if fx:
        dims.append(-1)
    if len(dims) > 0:
        t = torch.flip(t, dims)
    return t


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = nn.Conv2d(cin, SKIP, 1)

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        f = torch.cat([x, F.relu(self.skip(obs))], dim=1)
        return x, f


class WindowDoneHead(nn.Module):
    def __init__(self, c, spec):
        super().__init__()
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        K = max(max(a, b) for a, b in fps)
        self.nt = len(fps)
        self.frames = frame_list(int(spec.Y), int(spec.X))
        nf = len(self.frames)
        self.proj = nn.Conv2d(c, HID, 1)
        self.win = nn.Conv2d(HID, HID, 2 * K - 1, padding=K - 1)
        self.out = nn.Conv2d(HID, nf * self.nt, 1)
        self.alpha = nn.Parameter(torch.zeros(4 * nf))

    def forward(self, f, d):
        h = F.relu(self.proj(f))
        h = F.relu(self.win(h))
        m = self.out(h)
        T = self.nt
        cands = []
        for i, fr in enumerate(self.frames):
            mi = to_frame(m[:, i * T:(i + 1) * T], fr)
            tm = mi.flatten(1)
            am = mi.flatten(2).transpose(1, 2).flatten(1)
            for p in (tm, am):
                cands.append(torch.cat([p, d], dim=1))
                cands.append(torch.cat([d, p], dim=1))
        L = torch.stack(cands, dim=0)
        w = torch.softmax(SCALE * self.alpha, dim=0)
        return (L * w[:, None, None]).sum(0)


class CellAttnHead(nn.Module):
    def __init__(self, c, spec):
        super().__init__()
        self.N = int(spec.Y) * int(spec.X)
        self.A = int(spec.n_actions)
        self.proj = nn.Conv2d(c, HEADS * DH, 1)
        self.q = nn.Parameter(0.1 * torch.randn(HEADS, self.A, self.N))
        self.inp = nn.Linear(HEADS * DH, EMB)
        self.emb = nn.Parameter(0.5 * torch.randn(self.A, EMB))
        self.gl = nn.Linear(GH, EMB)
        self.out = nn.Linear(EMB, 1)

    def forward(self, f, g):
        B = f.shape[0]
        v = self.proj(f).reshape(B, HEADS, DH, self.N)
        att = torch.softmax(QSCALE * self.q, dim=-1)
        r = torch.einsum('han,bhdn->bahd', att, v).reshape(B, self.A, HEADS * DH)
        h = F.relu(self.inp(r) + self.emb + self.gl(g).unsqueeze(1))
        return self.out(h).squeeze(-1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec.in_channels)
        self.whead = WindowDoneHead(WIDTH + SKIP, spec)
        self.ahead = CellAttnHead(WIDTH + SKIP, spec)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.whead.out.weight, 0.01)
        nn.init.orthogonal_(self.ahead.out.weight, 0.01)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.orthogonal_(self.value_head.weight, 1.0)

    def forward(self, obs, mask):
        x, f = self.trunk(obs)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        d = self.done_head(g)
        logits = self.whead(f, d) + self.ahead(f, g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
