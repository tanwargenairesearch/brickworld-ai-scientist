import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

# layout = (order, transpose, flip_y, flip_x); order 0 type-major, 1 type-minor
LAYOUTS = [(0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
SHIFT_LAYOUTS = [(0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
OBS_SYMS = [(0, 0, 0)]
HEAD_K = 3
STEM = 64
WIDTH = 40
HEAD = 32
GH = 64


def to_grid(v, lay, T, Y, X):
    o, tr, fy, fx = lay
    B = v.shape[0]
    if o == 0:
        g = v.reshape(B, T, X, Y).transpose(2, 3) if tr else v.reshape(B, T, Y, X)
    else:
        g = v.reshape(B, X, Y, T).permute(0, 3, 2, 1) if tr else v.reshape(B, Y, X, T).permute(0, 3, 1, 2)
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    return g


def from_grid(g, lay):
    o, tr, fy, fx = lay
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    if o == 0:
        if tr:
            g = g.transpose(2, 3)
    else:
        g = g.permute(0, 3, 2, 1) if tr else g.permute(0, 2, 3, 1)
    return g.reshape(g.shape[0], -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y, self.X, self.T = int(spec.Y), int(spec.X), int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.ok = self.n_place == self.T * self.Y * self.X
        self.obs_syms = [s for s in OBS_SYMS if not (s[0] and self.Y != self.X)]
        self.lays = list(LAYOUTS) if self.ok else []
        self.slays = list(SHIFT_LAYOUTS) if self.ok else []
        cin = spec.in_channels * len(self.obs_syms) + 2 + self.T * (len(self.lays) + len(self.slays))
        if not self.ok:
            self.memb = nn.Linear(self.n_place, 8 * self.Y * self.X)
            cin += 8
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hk = nn.Conv2d(WIDTH, HEAD, HEAD_K, padding=HEAD_K // 2)
        self.hs = nn.Conv2d(WIDTH, HEAD, 1)
        if self.ok:
            self.out = nn.Conv2d(HEAD, self.T * len(self.lays), 1)
            if self.slays:
                self.sout = nn.Conv2d(HEAD, self.T * len(self.slays), 1)
        else:
            self.fo = nn.Conv2d(HEAD, 4, 1)
            self.fl = nn.Linear(4 * self.Y * self.X, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done = nn.Linear(GH, 1)
        if self.slays:
            self.done2 = nn.Linear(GH, 1)
        self.value = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T = self.Y, self.X, self.T
        m = mask.float()
        mp = m[:, :self.n_place]
        feats = []
        for tr, fy, fx in self.obs_syms:
            g = obs
            if tr:
                g = g.transpose(2, 3)
            if fy:
                g = g.flip(2)
            if fx:
                g = g.flip(3)
            feats.append(g)
        for lay in self.lays:
            feats.append(to_grid(mp, lay, T, Y, X))
        if self.slays:
            ms = m[:, 1:self.n_place + 1]
            for lay in self.slays:
                feats.append(to_grid(ms, lay, T, Y, X))
        if not self.ok:
            feats.append(self.memb(mp).view(B, 8, Y, X))
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        feats.append(yy)
        feats.append(xx)
        x = self.stem(torch.cat(feats, 1))
        x = F.relu(x + self.r1(x))
        x = F.relu(x + self.r2(x))
        h = F.relu(self.hk(x) + self.hs(x))
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))
        if self.ok:
            o = self.out(h)
            place = 0
            for i, lay in enumerate(self.lays):
                place = place + from_grid(o[:, i * T:(i + 1) * T], lay)
        else:
            place = self.fl(F.relu(self.fo(h)).flatten(1))
        logits = torch.cat([place, self.done(g)], 1)
        if self.slays:
            o2 = self.sout(h)
            ps = 0
            for i, lay in enumerate(self.slays):
                ps = ps + from_grid(o2[:, i * T:(i + 1) * T], lay)
            logits = logits + torch.cat([self.done2(g), ps], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
