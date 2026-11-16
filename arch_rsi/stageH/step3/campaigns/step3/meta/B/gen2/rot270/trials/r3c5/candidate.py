import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

LAYOUTS = [(True, False, True)]
OBS_SYM = True
FLIPS_ONLY = True
STEM = 64
WIDTH = 40
HEAD = 32
HID = 64


def apply_layout(t, lay):
    tr, fy, fx = lay
    if tr:
        t = t.transpose(-1, -2)
    if fy:
        t = t.flip(-2)
    if fx:
        t = t.flip(-1)
    return t


def invert_layout(t, lay):
    tr, fy, fx = lay
    if fx:
        t = t.flip(-1)
    if fy:
        t = t.flip(-2)
    if tr:
        t = t.transpose(-1, -2)
    return t


def obs_views(obs, square):
    views = [obs, obs.flip(-2), obs.flip(-1), obs.flip(-2).flip(-1)]
    if square and not FLIPS_ONLY:
        ot = obs.transpose(-1, -2)
        views = views + [ot, ot.flip(-2), ot.flip(-1), ot.flip(-2).flip(-1)]
    return torch.cat(views, 1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        hw = self.Y * self.X
        self.K = (self.n_place + hw - 1) // hw
        self.pad = self.K * hw - self.n_place
        self.square = self.Y == self.X
        lays = []
        for lay in LAYOUTS:
            if lay[0] and not self.square:
                lay = (False, lay[1], lay[2])
            if lay not in lays:
                lays.append(lay)
        self.layouts = lays
        L = len(lays)
        if OBS_SYM:
            nv = 8 if (self.square and not FLIPS_ONLY) else 4
        else:
            nv = 1
        obs_in = spec.in_channels * nv + 2
        mask_in = self.K * L
        self.stem = nn.Sequential(nn.Conv2d(obs_in + mask_in, STEM, 1), nn.ReLU(),
                                  nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        self.hout = nn.Conv2d(HEAD, self.K * L, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, HID), nn.ReLU())
        self.done_head = nn.Linear(HID, 1)
        self.value_head = nn.Linear(HID, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K = self.Y, self.X, self.K
        m = mask[:, :self.n_place].float()
        if self.pad > 0:
            m = F.pad(m, (0, self.pad))
        m = m.view(B, K, Y, X)
        mk = torch.cat([apply_layout(m, lay) for lay in self.layouts], 1)
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device).view(1, 1, 1, X).expand(B, 1, Y, X)
        o = obs_views(obs, self.square) if OBS_SYM else obs
        o = torch.cat([o, ys, xs], 1)
        x = self.stem(torch.cat([o, mk], 1))
        x = F.relu(x + self.r1(x))
        f = F.relu(x + self.r2(x))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = F.relu(self.h3(f) + self.h1(f))
        maps = self.hout(h).view(B, len(self.layouts), K, Y, X)
        place = 0
        for i, lay in enumerate(self.layouts):
            place = place + invert_layout(maps[:, i], lay)
        place = place.reshape(B, K * Y * X)[:, :self.n_place]
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
