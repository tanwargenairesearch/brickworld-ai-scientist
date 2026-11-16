import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

# (transpose, yflip, xflip)
LAYOUTS = [(False, False, False), (True, False, False), (False, True, False), (False, False, True)]
OBS_SYM = False
USE_LN = False
STEM = 64
WIDTH = 40
HEAD = 32
GLOB = 64


def fwd_t(x, t):
    tr, fy, fx = t
    if tr:
        x = x.transpose(-2, -1)
    if fy:
        x = x.flip(-2)
    if fx:
        x = x.flip(-1)
    return x


def inv_t(x, t):
    tr, fy, fx = t
    if fx:
        x = x.flip(-1)
    if fy:
        x = x.flip(-2)
    if tr:
        x = x.transpose(-2, -1)
    return x


def all_syms(square):
    out = []
    for tr in ([False, True] if square else [False]):
        for fy in [False, True]:
            for fx in [False, True]:
                out.append((tr, fy, fx))
    return out


class CLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y, self.X = int(spec.Y), int(spec.X)
        self.n_place = int(spec.n_place)
        sq = self.Y == self.X
        self.layouts = [t for t in LAYOUTS if sq or not t[0]]
        if len(self.layouts) == 0:
            self.layouts = [(False, False, False)]
        self.syms = all_syms(sq) if OBS_SYM else [(False, False, False)]
        cells = self.Y * self.X
        self.K = self.n_place // cells
        self.exact = self.K > 0 and self.K * cells == self.n_place
        L = len(self.layouts)
        cin_obs = spec.in_channels * len(self.syms)
        if self.exact:
            cin_mask = self.K * L
        else:
            cin_mask = 8
            self.mask_embed = nn.Linear(self.n_place, 8 * cells)
        if USE_LN:
            self.obs_stem = nn.Conv2d(cin_obs + 2, STEM, 1)
            self.obs_ln = CLN(STEM)
            self.mask_stem = nn.Conv2d(cin_mask, STEM, 1)
            self.mask_ln = CLN(STEM)
        else:
            self.stem = nn.Conv2d(cin_obs + cin_mask + 2, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.ln0 = CLN(WIDTH) if USE_LN else nn.Identity()
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.ln1 = CLN(WIDTH) if USE_LN else nn.Identity()
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.ln2 = CLN(WIDTH) if USE_LN else nn.Identity()
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        if self.exact:
            self.hout = nn.Conv2d(HEAD, self.K * L, 1)
        else:
            self.hout = nn.Conv2d(HEAD, 4, 1)
            self.hlin = nn.Linear(4 * cells, self.n_place)
        self.glob = nn.Linear(2 * WIDTH, GLOB)
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        if USE_LN:
            outs = [self.hout, self.done_head, self.value_head]
            if not self.exact:
                outs.append(self.hlin)
            for m in outs:
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        coords = torch.cat([ys, xs], 1)
        ob = torch.cat([fwd_t(obs, s) for s in self.syms], 1)
        mp = mask[:, :self.n_place].to(obs.dtype)
        if self.exact:
            m = mp.reshape(B, self.K, Y, X)
            mk = torch.cat([fwd_t(m, t) for t in self.layouts], 1)
        else:
            mk = self.mask_embed(mp).reshape(B, 8, Y, X)
        if USE_LN:
            s = F.relu(self.obs_ln(self.obs_stem(torch.cat([ob, coords], 1)))) + F.relu(self.mask_ln(self.mask_stem(mk)))
        else:
            s = F.relu(self.stem(torch.cat([ob, mk, coords], 1)))
        h = F.relu(self.ln0(self.stem2(s)))
        h = F.relu(self.ln1(h + self.r1(h)))
        h = F.relu(self.ln2(h + self.r2(h)))
        e = F.relu(self.h3(h) + self.h1(h))
        o = self.hout(e)
        if self.exact:
            K = self.K
            place = None
            for i, t in enumerate(self.layouts):
                p = inv_t(o[:, i * K:(i + 1) * K], t).reshape(B, -1)
                place = p if place is None else place + p
        else:
            place = self.hlin(o.flatten(1))
        g = F.relu(self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)))
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
