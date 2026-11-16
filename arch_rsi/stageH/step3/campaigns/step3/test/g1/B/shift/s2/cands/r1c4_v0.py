import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

LAYOUTS = ((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (0, 0, 1, 1),
           (0, 1, 0, 0), (0, 1, 1, 0), (0, 1, 0, 1), (0, 1, 1, 1),
           (1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1), (1, 0, 1, 1),
           (1, 1, 0, 0), (1, 1, 1, 0), (1, 1, 0, 1), (1, 1, 1, 1))
OBS_LAYOUTS = ((0, 0, 0),)
HEAD_K = 3
DONE_SHIFT = True
DENSE = False
STEM = 64
WIDTH = 40
HPROJ = 32
MEMB = 4


def lay_in(m, lay, T, Y, X):
    minor, trans, fy, fx = lay
    B = m.shape[0]
    if trans:
        s1, s2 = X, Y
    else:
        s1, s2 = Y, X
    if minor:
        m = m.reshape(B, s1, s2, T).permute(0, 3, 1, 2)
    else:
        m = m.reshape(B, T, s1, s2)
    if trans:
        m = m.transpose(2, 3)
    if fy:
        m = m.flip(2)
    if fx:
        m = m.flip(3)
    return m


def lay_out(L, lay):
    minor, trans, fy, fx = lay
    if fy:
        L = L.flip(2)
    if fx:
        L = L.flip(3)
    if trans:
        L = L.transpose(2, 3)
    if minor:
        L = L.permute(0, 2, 3, 1)
    return L.reshape(L.shape[0], -1)


def obs_lay(o, lay, Y, X):
    trans, fy, fx = lay
    B, C = o.shape[0], o.shape[1]
    if trans:
        o = o.reshape(B, C, X, Y).transpose(2, 3)
    if fy:
        o = o.flip(2)
    if fx:
        o = o.flip(3)
    return o


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.conv(x))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        T, Y, X = spec.n_types, spec.Y, spec.X
        self.T, self.Y, self.X = T, Y, X
        self.n_place = spec.n_place
        self.n_actions = spec.n_actions
        self.ok = spec.n_place == T * Y * X
        self.layouts = tuple(LAYOUTS) if self.ok else ()
        self.shift = DONE_SHIFT and self.ok and spec.n_actions == spec.n_place + 1
        self.dense = DENSE or (not self.ok)
        nsets = 2 if self.shift else 1
        in_c = len(OBS_LAYOUTS) * spec.in_channels + nsets * len(self.layouts) * T + 2
        if self.dense:
            self.mask_emb = nn.Sequential(nn.Linear(spec.n_actions, 64), nn.Linear(64, MEMB * Y * X))
            in_c += MEMB
        self.stem = nn.Sequential(nn.Conv2d(in_c, STEM, 1), nn.ReLU(),
                                  nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.res = nn.Sequential(Res(WIDTH), Res(WIDTH))
        self.hconv = nn.Conv2d(WIDTH, HPROJ, HEAD_K, padding=HEAD_K // 2)
        self.hskip = nn.Conv2d(WIDTH, HPROJ, 1)
        if len(self.layouts) > 0:
            self.out = nn.Conv2d(HPROJ, nsets * len(self.layouts) * T, 1)
        if self.dense:
            self.dproj = nn.Conv2d(HPROJ, 2, 1)
            self.dout = nn.Sequential(nn.Linear(2 * Y * X, 256), nn.ReLU(), nn.Linear(256, spec.n_actions))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        if self.shift:
            self.done_head2 = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        T, Y, X = self.T, self.Y, self.X
        m = mask.to(obs.dtype)
        feats = [obs_lay(obs, lay, Y, X) for lay in OBS_LAYOUTS]
        for lay in self.layouts:
            feats.append(lay_in(m[:, :self.n_place], lay, T, Y, X))
        if self.shift:
            for lay in self.layouts:
                feats.append(lay_in(m[:, 1:self.n_place + 1], lay, T, Y, X))
        if self.dense:
            feats.append(self.mask_emb(m).reshape(B, MEMB, Y, X))
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        feats.append(yy)
        feats.append(xx)
        f = self.res(self.stem(torch.cat(feats, 1)))
        h = F.relu(self.hconv(f) + self.hskip(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        done = self.done_head(g)
        if len(self.layouts) > 0:
            o = self.out(h)
            n = len(self.layouts)
            place = 0
            for k, lay in enumerate(self.layouts):
                place = place + lay_out(o[:, k * T:(k + 1) * T], lay)
            logits = torch.cat([place, done], 1)
            if self.shift:
                place2 = 0
                for k, lay in enumerate(self.layouts):
                    j = n + k
                    place2 = place2 + lay_out(o[:, j * T:(j + 1) * T], lay)
                logits = logits + torch.cat([self.done_head2(g), place2], 1)
        else:
            logits = torch.zeros(B, self.n_actions, device=obs.device, dtype=obs.dtype)
        if self.dense:
            logits = logits + self.dout(self.dproj(h).flatten(1))
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
