import torch
import torch.nn as nn
import torch.nn.functional as F
import math

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

WIDTH = 40
SKIP = 16
HID = 20
GHID = 64
SCALE = 5.0


def frame_list(Y, X):
    out = []
    for tr in ([0, 1] if Y == X else [0]):
        for fy in (0, 1):
            for fx in (0, 1):
                out.append((tr, fy, fx))
    return out


def to_orig(tr, fy, fx, Yf, Xf, yy, xx):
    a = Yf - 1 - yy if fy else yy
    b = Xf - 1 - xx if fx else xx
    if tr:
        return b, a
    return a, b


def init_net(m):
    for mod in m.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(mod.weight, math.sqrt(2))
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)


class Trunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = spec.in_channels
        self.stem = nn.Sequential(nn.Conv2d(C, 64, 1), nn.ReLU(), nn.Conv2d(64, 48, 1), nn.ReLU())
        self.c1 = nn.Conv2d(48, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = nn.Conv2d(C, SKIP, 1)
        self.out_ch = WIDTH + SKIP
        self.glob = nn.Sequential(nn.Linear(2 * self.out_ch, GHID), nn.ReLU())

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        f = torch.cat([x, self.skip(obs)], dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        return f, g


class WindowHead(nn.Module):
    def __init__(self, spec, cin):
        super().__init__()
        Y, X = int(spec.Y), int(spec.X)
        T = int(spec.n_types)
        K = max(max(int(a), int(b)) for a, b in spec.footprints)
        frames = frame_list(Y, X)
        nf = len(frames)
        YX = Y * X
        flat = []
        for gi, (tr, fy, fx) in enumerate(frames):
            Yf, Xf = (X, Y) if tr else (Y, X)
            for t in range(T):
                for yy in range(Yf):
                    for xx in range(Xf):
                        oy, ox = to_orig(tr, fy, fx, Yf, Xf, yy, xx)
                        flat.append((gi * T + t) * YX + oy * X + ox)
        self.register_buffer("gidx", torch.tensor(flat, dtype=torch.long))
        self.nf = nf
        self.T = T
        self.YX = YX
        self.win = nn.Conv2d(cin, HID, 2 * K - 1, padding=K - 1)
        self.read = nn.Conv2d(HID, nf * T, 1)

    def forward(self, f):
        B = f.shape[0]
        h = F.relu(self.win(f))
        M = self.read(h).flatten(1)
        L = M[:, self.gidx].view(B, self.nf, self.T, self.YX)
        Lt = L.flatten(2)
        La = L.transpose(2, 3).flatten(2)
        return torch.cat([Lt, La], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec)
        self.head = WindowHead(spec, self.trunk.out_ch)
        self.mixw = nn.Parameter(torch.zeros(2 * self.head.nf))
        self.done_head = nn.Linear(GHID, 1)
        self.value_head = nn.Linear(GHID, 1)
        self.dense = nn.Linear(GHID, int(spec.n_place))
        init_net(self)
        for m in [self.done_head, self.dense, self.head.read]:
            nn.init.orthogonal_(m.weight, 0.01)
        nn.init.orthogonal_(self.value_head.weight, 1.0)

    def forward(self, obs, mask):
        f, g = self.trunk(obs)
        L = self.head(f)
        w = torch.softmax(SCALE * self.mixw, 0)
        place = torch.einsum('f,bfn->bn', w, L) + self.dense(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
