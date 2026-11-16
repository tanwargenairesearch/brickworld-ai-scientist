import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

LAYOUTS = [(0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
TYPE_MINOR = False
DENSE = False
SEP_CRITIC = False
STEM = 64
WIDTH = 40
HEADW = 32
MSTEM = 32
DHID = 192


def fwd_lay(v, t):
    tr, fy, fx = t
    if tr:
        v = v.transpose(-1, -2)
    if fy:
        v = v.flip(-2)
    if fx:
        v = v.flip(-1)
    return v


def inv_lay(v, t):
    tr, fy, fx = t
    if fx:
        v = v.flip(-1)
    if fy:
        v = v.flip(-2)
    if tr:
        v = v.transpose(-1, -2)
    return v


class CellLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class GBlock(nn.Module):
    def __init__(self, c, dil):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=dil, dilation=dil)
        self.g = nn.Linear(2 * c, c)
        self.ln = CellLN(c)

    def forward(self, h):
        g = self.g(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))
        return F.relu(self.ln(h + self.conv(h) + g[:, :, None, None]))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        hw = self.Y * self.X
        self.K = (self.n_place + hw - 1) // hw
        sq = self.Y == self.X
        self.lays = [l for l in LAYOUTS if sq or l[0] == 0]
        if len(self.lays) == 0:
            self.lays = [(0, 1, 1)]
        self.syms = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1) if sq or a == 0]
        L = len(self.lays)
        cin_obs = spec.in_channels * len(self.syms) + 2
        cin_mask = self.K * L
        self.obs_stem = nn.Sequential(nn.Conv2d(cin_obs, STEM, 1), CellLN(STEM), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1))
        self.mask_stem = nn.Sequential(nn.Conv2d(cin_mask, MSTEM, 1), CellLN(MSTEM), nn.ReLU(), nn.Conv2d(MSTEM, WIDTH, 1))
        self.stem_ln = CellLN(WIDTH)
        self.b1 = GBlock(WIDTH, 1)
        self.b2 = GBlock(WIDTH, 2)
        self.h3 = nn.Conv2d(WIDTH, HEADW, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEADW, 1)
        self.hout = nn.Conv2d(HEADW, self.K * L, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        outs = [self.hout, self.done_head]
        if DENSE:
            self.d_obs = nn.Linear(spec.in_channels * hw, DHID)
            self.d_mask = nn.Linear(self.n_actions, DHID)
            self.d_ln = nn.LayerNorm(DHID)
            self.d_out = nn.Linear(DHID, self.n_actions)
            outs.append(self.d_out)
        if SEP_CRITIC:
            self.c_stem = nn.Sequential(nn.Conv2d(spec.in_channels + 2, 32, 1), CellLN(32), nn.ReLU())
            self.c_conv = nn.Conv2d(32, 32, 3, padding=1)
            self.c_fc = nn.Linear(64, 64)
            self.c_out = nn.Linear(64, 1)
        for m in outs:
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.zeros_(m.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K = self.Y, self.X, self.K
        hw = Y * X
        mf = mask.to(obs.dtype)
        mp = mf[:, :self.n_place]
        if K * hw > self.n_place:
            mp = F.pad(mp, (0, K * hw - self.n_place))
        if TYPE_MINOR:
            mp = mp.reshape(B, Y, X, K).permute(0, 3, 1, 2)
        else:
            mp = mp.reshape(B, K, Y, X)
        mviews = torch.cat([fwd_lay(mp, l) for l in self.lays], 1)
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        oin = torch.cat([fwd_lay(obs, s) for s in self.syms] + [ys, xs], 1)
        h = F.relu(self.stem_ln(self.obs_stem(oin) + self.mask_stem(mviews)))
        h = self.b1(h)
        h = self.b2(h)
        a = F.relu(self.h3(h) + self.h1(h))
        o = self.hout(a)
        place = None
        for i, l in enumerate(self.lays):
            m = inv_lay(o[:, i * K:(i + 1) * K], l)
            if TYPE_MINOR:
                m = m.permute(0, 2, 3, 1).reshape(B, K * hw)
            else:
                m = m.reshape(B, K * hw)
            place = m if place is None else place + m
        place = place[:, :self.n_place]
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))
        logits = torch.cat([place, self.done_head(g)], 1)
        if DENSE:
            d = F.relu(self.d_ln(self.d_obs(obs.flatten(1)) + self.d_mask(mf)))
            logits = logits + self.d_out(d)
        if SEP_CRITIC:
            c = self.c_stem(torch.cat([obs, ys, xs], 1))
            c = F.relu(c + self.c_conv(c))
            v = self.c_out(F.relu(self.c_fc(torch.cat([c.mean((2, 3)), c.amax((2, 3))], 1))))
        else:
            v = self.value_head(g)
        return logits, v.squeeze(1)


def build_policy(spec):
    return Net(spec)
