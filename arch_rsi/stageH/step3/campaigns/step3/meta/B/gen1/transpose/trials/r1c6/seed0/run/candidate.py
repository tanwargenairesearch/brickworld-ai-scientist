import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

STEM = 64
WIDTH = 40
HEAD = 32
HEAD_K = 5
WIDE = True
DONE_BOTH = True
OBS_SYM = True


def layout_list(wide):
    if not wide:
        return [(False, False, False, False), (True, False, False, False),
                (False, True, False, False), (False, False, True, False)]
    out = []
    for minor in (False, True):
        for swap in (False, True):
            for fy in (False, True):
                for fx in (False, True):
                    out.append((swap, fy, fx, minor))
    return out


def layout_perm(T, Y, X, swap, fy, fx, minor):
    t = torch.arange(T).view(T, 1, 1).expand(T, Y, X)
    y = torch.arange(Y).view(1, Y, 1).expand(T, Y, X)
    x = torch.arange(X).view(1, 1, X).expand(T, Y, X)
    if fy:
        y = (Y - 1) - y
    if fx:
        x = (X - 1) - x
    if swap:
        sp = x * Y + y
    else:
        sp = y * X + x
    if minor:
        idx = sp * T + t
    else:
        idx = t * (Y * X) + sp
    return idx.reshape(-1)


def obs_variants(obs, square):
    outs = []
    for swap in ((False, True) if square else (False,)):
        for fy in (False, True):
            for fx in (False, True):
                o = obs
                if swap:
                    o = o.transpose(2, 3)
                if fy:
                    o = o.flip(2)
                if fx:
                    o = o.flip(3)
                outs.append(o)
    return outs


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        T, Y, X = int(spec.n_types), int(spec.Y), int(spec.X)
        self.T, self.Y, self.X = T, Y, X
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.grid_ok = (self.n_place == T * Y * X) and (self.n_actions == self.n_place + 1)
        self.square = (Y == X)
        self.obs_sym = OBS_SYM
        self.done_both = DONE_BOTH and self.grid_ok
        n_obs = (8 if self.square else 4) if OBS_SYM else 1
        C = int(spec.in_channels)
        yy = torch.linspace(-1.0, 1.0, Y).view(1, 1, Y, 1).expand(1, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X).view(1, 1, 1, X).expand(1, 1, Y, X)
        self.register_buffer("coords", torch.cat([yy, xx], dim=1).contiguous())
        in_ch = C * n_obs + 2
        self.n_lay = 1
        self.n_off = 1
        self.scale = 1.0
        if self.grid_ok:
            perms = [layout_perm(T, Y, X, *l) for l in layout_list(WIDE)]
            self.n_lay = len(perms)
            self.n_off = 2 if self.done_both else 1
            self.register_buffer("perm_all", torch.cat(perms))
            self.register_buffer("inv_all", torch.stack([torch.argsort(p) for p in perms] * self.n_off))
            in_ch += self.n_lay * self.n_off * T
            self.scale = 2.0 / math.sqrt(self.n_lay)
        self.s1 = nn.Conv2d(in_ch, STEM, 1)
        self.s2 = nn.Conv2d(STEM, WIDTH, 1)
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        if self.grid_ok:
            self.hk = nn.Conv2d(WIDTH, HEAD, HEAD_K, padding=HEAD_K // 2)
            self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
            self.hout = nn.Conv2d(HEAD, self.n_lay * self.n_off * T, 1)
            if self.done_both:
                self.done_head2 = nn.Linear(64, 1)
        else:
            self.membed = nn.Linear(self.n_actions, STEM)
            self.fb_out = nn.Linear(64, self.n_actions)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T = self.Y, self.X, self.T
        parts = obs_variants(obs, self.square) if self.obs_sym else [obs]
        parts = parts + [self.coords.expand(B, -1, -1, -1).to(obs.dtype)]
        m = mask.to(obs.dtype)
        if self.grid_ok:
            n = self.n_place
            parts.append(m[:, :n].index_select(1, self.perm_all).view(B, self.n_lay * T, Y, X))
            if self.done_both:
                parts.append(m[:, 1:1 + n].index_select(1, self.perm_all).view(B, self.n_lay * T, Y, X))
        s = self.s1(torch.cat(parts, dim=1))
        if not self.grid_ok:
            s = s + self.membed(m).view(B, -1, 1, 1)
        s = F.relu(self.s2(F.relu(s)))
        x = F.relu(s + self.r1(s))
        f = F.relu(x + self.r2(x))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        done = self.done_head(g)
        value = self.value_head(g).squeeze(1)
        if not self.grid_ok:
            return self.fb_out(g), value
        h = F.relu(self.hk(f) + self.h1(f))
        maps = self.hout(h).reshape(B, self.n_lay * self.n_off, -1)
        idx = self.inv_all.unsqueeze(0).expand(B, -1, -1)
        lay = torch.gather(maps, 2, idx)
        L = self.n_lay
        place0 = lay[:, :L].sum(1) * self.scale
        logits = torch.cat([place0, done], dim=1)
        if self.done_both:
            place1 = lay[:, L:].sum(1) * self.scale
            logits = logits + torch.cat([self.done_head2(g), place1], dim=1)
        return logits, value


def build_policy(spec):
    return Net(spec)
