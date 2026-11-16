import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 32, "n_minibatches": 8,
          "ppo_epochs": 2, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
WIDTH = 40
HEAD = 32
HEAD_K = 3
EMB = 8
# layout = (transpose, flip_y, flip_x)
LAYOUTS = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
           (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
POOL = 0
WINDOWS = [(1, 2), (2, 1), (2, 2)]
GLOBAL_CTX = False
REMAP = False


def to_grid(m, lay, K, Y, X):
    tr, fy, fx = lay
    B = m.shape[0]
    if tr:
        g = m.reshape(B, K, X, Y).transpose(2, 3)
    else:
        g = m.reshape(B, K, Y, X)
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    return g


def from_grid(g, lay):
    tr, fy, fx = lay
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    if tr:
        g = g.transpose(2, 3)
    return g.reshape(g.shape[0], -1)


def window_feats(p, wins):
    out = [p]
    for dy, dx in wins:
        for pad in ((0, dx - 1, 0, dy - 1), (dx - 1, 0, dy - 1, 0)):
            q = F.pad(p, pad)
            out.append(F.max_pool2d(q, (dy, dx), stride=1))
            out.append(-F.max_pool2d(-q, (dy, dx), stride=1))
    return torch.cat(out, 1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.n_extra = self.n_actions - self.n_place
        YX = self.Y * self.X
        self.grid = self.n_place > 0 and self.n_place % YX == 0
        self.K = self.n_place // YX
        self.lays = [tuple(l) for l in LAYOUTS]
        cin = self.C + 2
        if REMAP:
            self.remap = nn.Linear(YX, YX, bias=False)
            nn.init.orthogonal_(self.remap.weight)
            cin += self.C
        if self.grid:
            cin += len(self.lays) * self.K
        else:
            self.mask_embed = nn.Linear(self.n_actions, EMB * YX)
            cin += EMB
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(),
                                  nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        if GLOBAL_CTX:
            self.g1 = nn.Linear(2 * WIDTH, WIDTH)
            self.g2 = nn.Linear(2 * WIDTH, WIDTH)
        self.hconv = nn.Conv2d(WIDTH, HEAD, HEAD_K, padding=HEAD_K // 2)
        self.hskip = nn.Conv2d(WIDTH, HEAD, 1)
        if POOL > 0:
            self.pproj = nn.Conv2d(WIDTH, POOL, 1)
            self.hpool = nn.Conv2d(POOL * (1 + 4 * len(WINDOWS)), HEAD, 1)
        if self.grid:
            self.hout = nn.Conv2d(HEAD, len(self.lays) * self.K, 1)
        else:
            self.hred = nn.Conv2d(HEAD, EMB, 1)
            self.hlin = nn.Linear(EMB * YX, self.n_actions)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, max(self.n_extra, 1))
        self.value_head = nn.Linear(64, 1)

    def ctx(self, s, lin):
        g = torch.cat([s.mean((2, 3)), s.amax((2, 3))], 1)
        return s + F.relu(lin(g))[:, :, None, None]

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        feats = [obs]
        if REMAP:
            feats.append(self.remap(obs.reshape(B, self.C, Y * X)).reshape(B, self.C, Y, X))
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        feats.append(yy)
        feats.append(xx)
        mf = mask.to(obs.dtype)
        if self.grid:
            seg = mf[:, :self.n_place]
            for lay in self.lays:
                feats.append(to_grid(seg, lay, self.K, Y, X))
        else:
            feats.append(self.mask_embed(mf).view(B, -1, Y, X))
        s = self.stem(torch.cat(feats, 1))
        s = F.relu(s + self.c1(s))
        if GLOBAL_CTX:
            s = self.ctx(s, self.g1)
        f = F.relu(s + self.c2(s))
        if GLOBAL_CTX:
            f = self.ctx(f, self.g2)
        h = self.hconv(f) + self.hskip(f)
        if POOL > 0:
            h = h + self.hpool(window_feats(self.pproj(f), WINDOWS))
        h = F.relu(h)
        if self.grid:
            maps = self.hout(h)
            acc = None
            for i, lay in enumerate(self.lays):
                part = from_grid(maps[:, i * self.K:(i + 1) * self.K], lay)
                acc = part if acc is None else acc + part
            logits = F.pad(acc, (0, self.n_extra))
        else:
            logits = self.hlin(self.hred(h).flatten(1))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        d = self.done_head(g)[:, :self.n_extra]
        logits = logits + F.pad(d, (self.n_place, 0))
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
