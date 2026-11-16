import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}
LAYOUTS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
ORDER = "yxt"
OBS_VIEWS = ("std",)
OBS_SYM = False
DONE_FIRST = False
USE_LN = False
GCTX = False
DILATE = False
STEM = 64
WIDTH = 40
HEADC = 32
HID = 64
MASKC = 8


def lay(t, L):
    tr, fy, fx = L
    if tr:
        t = t.transpose(2, 3)
    if fy:
        t = t.flip(2)
    if fx:
        t = t.flip(3)
    return t


def unlay(t, L):
    tr, fy, fx = L
    if fx:
        t = t.flip(3)
    if fy:
        t = t.flip(2)
    if tr:
        t = t.transpose(2, 3)
    return t


class ChanLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def norm(c):
    if USE_LN:
        return ChanLN(c)
    return nn.Identity()


class Block(nn.Module):
    def __init__(self, c, dil):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=dil, dilation=dil)
        self.norm = norm(c)
        self.use_g = GCTX
        if GCTX:
            self.gctx = nn.Linear(2 * c, c)

    def forward(self, x):
        y = self.conv(x)
        if self.use_g:
            g = self.gctx(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
            y = y + g[:, :, None, None]
        return F.relu(self.norm(x + y))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        cells = self.Y * self.X
        self.K = max(1, self.n_place // cells)
        sq = self.Y == self.X
        self.layouts = [L for L in LAYOUTS if sq or L[0] == 0]
        if len(self.layouts) == 0:
            self.layouts = [(0, 0, 0)]
        if OBS_SYM:
            self.syms = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1) if sq or a == 0]
        else:
            self.syms = [(0, 0, 0)]
        n_views = 0
        for v in OBS_VIEWS:
            n_views += len(self.syms) if v == "std" else 1
        nL = len(self.layouts)
        self.mask_direct = self.K * cells == self.n_place
        obs_c = self.C * n_views + 2
        if self.mask_direct:
            mask_c = self.K * nL
        else:
            self.mask_embed = nn.Linear(self.n_place, MASKC * cells)
            mask_c = MASKC
        if USE_LN:
            self.obs_stem = nn.Sequential(nn.Conv2d(obs_c, STEM, 1), ChanLN(STEM), nn.ReLU())
            self.mask_stem = nn.Sequential(nn.Conv2d(mask_c, STEM, 1), ChanLN(STEM), nn.ReLU())
            self.stem2 = nn.Sequential(nn.Conv2d(STEM, WIDTH, 1), ChanLN(WIDTH), nn.ReLU())
        else:
            self.stem = nn.Sequential(nn.Conv2d(obs_c + mask_c, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.b1 = Block(WIDTH, 1)
        self.b2 = Block(WIDTH, 2 if DILATE else 1)
        self.hconv = nn.Conv2d(WIDTH, HEADC, 3, padding=1)
        self.hskip = nn.Conv2d(WIDTH, HEADC, 1)
        self.hout = nn.Conv2d(HEADC, self.K * nL, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, HID), nn.ReLU())
        self.done_head = nn.Linear(HID, 1)
        self.value_head = nn.Linear(HID, 1)
        if USE_LN:
            nn.init.orthogonal_(self.hout.weight, 0.01)
            nn.init.zeros_(self.hout.bias)
            nn.init.orthogonal_(self.done_head.weight, 0.01)
            nn.init.zeros_(self.done_head.bias)
            nn.init.orthogonal_(self.value_head.weight, 1.0)
            nn.init.zeros_(self.value_head.bias)

    def read_mask(self, mp):
        B = mp.shape[0]
        Y, X, K = self.Y, self.X, self.K
        if ORDER == "yxt":
            return mp.reshape(B, Y, X, K).permute(0, 3, 1, 2)
        if ORDER == "ytx":
            return mp.reshape(B, Y, K, X).permute(0, 2, 1, 3)
        return mp.reshape(B, K, Y, X)

    def write_out(self, q):
        B = q.shape[0]
        if ORDER == "yxt":
            q = q.permute(0, 2, 3, 1)
        elif ORDER == "ytx":
            q = q.permute(0, 2, 1, 3)
        return q.reshape(B, -1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K, C = self.Y, self.X, self.K, self.C
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        views = []
        for v in OBS_VIEWS:
            if v == "std":
                views += [lay(obs, s) for s in self.syms]
            elif v == "yxc":
                views.append(obs.reshape(B, Y, X, C).permute(0, 3, 1, 2))
            elif v == "ycx":
                views.append(obs.reshape(B, Y, C, X).permute(0, 2, 1, 3))
        o = torch.cat(views + [ys, xs], dim=1)
        if DONE_FIRST:
            mp = mask[:, 1:1 + self.n_place].to(obs.dtype)
        else:
            mp = mask[:, :self.n_place].to(obs.dtype)
        if self.mask_direct:
            m = self.read_mask(mp)
            m = torch.cat([lay(m, L) for L in self.layouts], dim=1)
        else:
            m = self.mask_embed(mp).view(B, MASKC, Y, X)
        if USE_LN:
            s = self.stem2(self.obs_stem(o) + self.mask_stem(m))
        else:
            s = self.stem(torch.cat([o, m], dim=1))
        f = self.b2(self.b1(s))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = F.relu(self.hconv(f) + self.hskip(f))
        maps = self.hout(h)
        place = None
        for i, L in enumerate(self.layouts):
            p = self.write_out(unlay(maps[:, i * K:(i + 1) * K], L))
            place = p if place is None else place + p
        n = place.shape[1]
        if n < self.n_place:
            place = F.pad(place, (0, self.n_place - n))
        elif n > self.n_place:
            place = place[:, :self.n_place]
        d = self.done_head(g)
        if DONE_FIRST:
            logits = torch.cat([d, place], dim=1)
        else:
            logits = torch.cat([place, d], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
