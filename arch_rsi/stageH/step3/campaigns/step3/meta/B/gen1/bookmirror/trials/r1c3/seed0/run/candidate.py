import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

LAYOUT_MODE = "ref4"
HEAD_K = 5
HEAD_W = 32
DONE_BOTH = False
OBS_SYM = False
STEM = 64
WIDTH = 40


def make_layouts(mode):
    if mode == "ref4":
        return [(0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
    out = []
    for minor in (0, 1):
        for tr in (0, 1):
            for fy in (0, 1):
                for fx in (0, 1):
                    out.append((minor, tr, fy, fx))
    return out


def lay(v, lt, K, Y, X):
    minor, tr, fy, fx = lt
    B = v.shape[0]
    a, b = (X, Y) if tr else (Y, X)
    if minor:
        m = v.reshape(B, a, b, K).permute(0, 3, 1, 2)
    else:
        m = v.reshape(B, K, a, b)
    if tr:
        m = m.transpose(2, 3)
    if fy:
        m = m.flip(2)
    if fx:
        m = m.flip(3)
    return m


def unlay(m, lt):
    minor, tr, fy, fx = lt
    if fy:
        m = m.flip(2)
    if fx:
        m = m.flip(3)
    if tr:
        m = m.transpose(2, 3)
    if minor:
        m = m.permute(0, 2, 3, 1)
    return m.reshape(m.shape[0], -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.N = int(spec.n_actions)
        self.P = self.N - 1
        YX = self.Y * self.X
        self.grid = (self.P % YX == 0)
        self.K = self.P // YX if self.grid else 0
        self.layouts = make_layouts(LAYOUT_MODE)
        self.nv = 2 if DONE_BOTH else 1
        syms = []
        if OBS_SYM:
            for tr in ((0, 1) if self.Y == self.X else (0,)):
                for fy in (0, 1):
                    for fx in (0, 1):
                        syms.append((tr, fy, fx))
        else:
            syms.append((0, 0, 0))
        self.osyms = syms
        cin = int(spec.in_channels) * len(syms) + 2
        if self.grid:
            cin += self.nv * len(self.layouts) * self.K
        else:
            self.mask_embed = nn.Linear(self.N, 8 * YX)
            cin += 8
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hconv = nn.Conv2d(WIDTH, HEAD_W, HEAD_K, padding=HEAD_K // 2)
        self.hskip = nn.Conv2d(WIDTH, HEAD_W, 1)
        if self.grid:
            self.hout = nn.Conv2d(HEAD_W, self.nv * len(self.layouts) * self.K, 1)
        else:
            self.hred = nn.Conv2d(HEAD_W, 8, 1)
            self.hlin = nn.Linear(8 * YX, self.nv * self.P)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, self.nv)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        feats = []
        for tr, fy, fx in self.osyms:
            o = obs
            if tr:
                o = o.transpose(2, 3)
            if fy:
                o = o.flip(2)
            if fx:
                o = o.flip(3)
            feats.append(o)
        m = mask.float()
        views = [m[:, :self.P]]
        if DONE_BOTH:
            views.append(m[:, 1:])
        if self.grid:
            for v in views:
                for lt in self.layouts:
                    feats.append(lay(v, lt, self.K, Y, X))
        else:
            feats.append(self.mask_embed(m).view(B, 8, Y, X))
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device).view(1, 1, 1, X).expand(B, 1, Y, X)
        feats.append(yy)
        feats.append(xx)
        s = self.stem(torch.cat(feats, dim=1))
        f = F.relu(s + self.r1(s))
        f = F.relu(f + self.r2(f))
        h = F.relu(self.hconv(f) + self.hskip(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        d = self.done_head(g)
        if self.grid:
            o = self.hout(h)
            L = len(self.layouts)
            K = self.K
            places = []
            for vi in range(self.nv):
                place = 0
                for li, lt in enumerate(self.layouts):
                    idx = (vi * L + li) * K
                    place = place + unlay(o[:, idx:idx + K], lt)
                places.append(place)
        else:
            pl = self.hlin(F.relu(self.hred(h)).flatten(1))
            places = [pl[:, vi * self.P:(vi + 1) * self.P] for vi in range(self.nv)]
        logits = torch.cat([places[0], d[:, 0:1]], dim=1)
        if self.nv > 1:
            logits = logits + torch.cat([d[:, 1:2], places[1]], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
