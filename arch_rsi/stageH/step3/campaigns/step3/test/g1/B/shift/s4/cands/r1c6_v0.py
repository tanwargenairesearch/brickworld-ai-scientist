import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

WIDE = True
HEAD_K = 5
OBS_VIEWS = True
GLOBAL_BIAS = True
STEM = 64
W = 40
HP = 32


def make_layouts(wide):
    if wide:
        return [(tm, tr, fy, fx) for tm in (True, False) for tr in (False, True)
                for fy in (False, True) for fx in (False, True)]
    return [(True, False, False, False), (True, True, False, False),
            (True, False, True, False), (True, False, False, True)]


def flip_dims(fy, fx):
    d = []
    if fy:
        d.append(2)
    if fx:
        d.append(3)
    return d


def to_grid(m, lay, T, Y, X):
    tm, tr, fy, fx = lay
    B = m.shape[0]
    A, C = (X, Y) if tr else (Y, X)
    if tm:
        g = m.reshape(B, T, A, C)
    else:
        g = m.reshape(B, A, C, T).permute(0, 3, 1, 2)
    if tr:
        g = g.transpose(2, 3)
    d = flip_dims(fy, fx)
    if d:
        g = g.flip(d)
    return g


def from_grid(g, lay):
    tm, tr, fy, fx = lay
    d = flip_dims(fy, fx)
    if d:
        g = g.flip(d)
    if tr:
        g = g.transpose(2, 3)
    if not tm:
        g = g.permute(0, 2, 3, 1)
    return g.reshape(g.shape[0], -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y, self.X, self.T = spec.Y, spec.X, spec.n_types
        self.n_place = spec.n_place
        self.ok = spec.n_place == self.T * self.Y * self.X
        self.lays = make_layouts(WIDE)
        self.square = self.Y == self.X
        nv = 1
        if OBS_VIEWS:
            nv = 4 if self.square else 3
        cin = spec.in_channels * nv + 2
        if self.ok:
            cin += self.T * len(self.lays)
        else:
            self.mask_emb = nn.Linear(spec.n_actions, 4 * self.Y * self.X)
            cin += 4
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(),
                                  nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.r1 = nn.Conv2d(W, W, 3, padding=1)
        self.r2 = nn.Conv2d(W, W, 3, padding=1)
        self.hconv = nn.Conv2d(W, HP, HEAD_K, padding=HEAD_K // 2)
        self.hskip = nn.Conv2d(W, HP, 1)
        if self.ok:
            self.outs = nn.Conv2d(HP, self.T * len(self.lays), 1)
        else:
            self.red = nn.Conv2d(HP, 8, 1)
            self.dense = nn.Linear(8 * self.Y * self.X, spec.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        if GLOBAL_BIAS:
            self.gbias = nn.Linear(64, spec.n_actions)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T = self.Y, self.X, self.T
        ins = [obs]
        if OBS_VIEWS:
            if self.square:
                ins.append(obs.transpose(2, 3))
            ins.append(obs.flip([2]))
            ins.append(obs.flip([3]))
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device).view(1, 1, 1, X).expand(B, 1, Y, X)
        ins += [yy, xx]
        mf = mask.float()
        if self.ok:
            mp = mf[:, :self.n_place]
            for lay in self.lays:
                ins.append(to_grid(mp, lay, T, Y, X))
        else:
            ins.append(self.mask_emb(mf).view(B, 4, Y, X))
        x = self.stem(torch.cat(ins, dim=1))
        x = F.relu(x + self.r1(x))
        f = F.relu(x + self.r2(x))
        h = F.relu(self.hconv(f) + self.hskip(f))
        if self.ok:
            maps = self.outs(h)
            place = None
            for k, lay in enumerate(self.lays):
                p = from_grid(maps[:, k * T:(k + 1) * T], lay)
                place = p if place is None else place + p
        else:
            place = self.dense(F.relu(self.red(h)).flatten(1))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        if GLOBAL_BIAS:
            logits = logits + self.gbias(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
