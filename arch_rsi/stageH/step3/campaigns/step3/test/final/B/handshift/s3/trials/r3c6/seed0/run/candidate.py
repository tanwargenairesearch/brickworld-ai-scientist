import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}
OBS_SYM = True
REINTERP = True
INST_NORM = False
TOKEN_MIX = False
DONE_FIRST = False
DENSE_MLP = False
STEM = 64
WIDTH = 40
HEAD = 32
GLOB = 64
MASK_EMB = 8
MLP_HID = 256


def sym(t, tr, fy, fx):
    if tr:
        t = t.transpose(-1, -2)
    if fy:
        t = t.flip(-2)
    if fx:
        t = t.flip(-1)
    return t


class ChanLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def pooled(x):
    return torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.ctx = nn.Linear(2 * c, c)
        self.norm = ChanLN(c)

    def forward(self, x):
        y = x + self.conv(x) + self.ctx(pooled(x))[:, :, None, None]
        return F.relu(self.norm(y))


class TokenMix(nn.Module):
    def __init__(self, c, n):
        super().__init__()
        self.mix = nn.Linear(n, n)
        self.norm = ChanLN(c)

    def forward(self, x):
        b, c, yy, xx = x.shape
        m = self.mix(x.reshape(b, c, yy * xx)).reshape(b, c, yy, xx)
        return F.relu(self.norm(x + m))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        cells = self.Y * self.X
        square = self.Y == self.X
        if OBS_SYM:
            syms = [(t, a, b) for t in (0, 1) for a in (0, 1) for b in (0, 1)]
            self.syms = [s for s in syms if square or not s[0]]
        else:
            self.syms = [(0, 0, 0)]
        n_views = len(self.syms) * (2 if REINTERP else 1)
        view_ch = self.C * n_views
        self.grid = (self.n_place % cells == 0)
        if self.grid:
            self.K = self.n_place // cells
            mask_in = self.K
        else:
            self.K = 1
            self.mask_embed = nn.Linear(self.n_place, cells * MASK_EMB)
            mask_in = MASK_EMB
        self.obs_stem = nn.Sequential(nn.Conv2d(view_ch + 2, STEM, 1), ChanLN(STEM), nn.ReLU())
        self.mask_stem = nn.Sequential(nn.Conv2d(mask_in, STEM, 1), ChanLN(STEM), nn.ReLU())
        if INST_NORM:
            self.dw = nn.Conv2d(view_ch, view_ch, 1, groups=view_ch)
            self.inst_stem = nn.Sequential(nn.Conv2d(view_ch, STEM, 1), ChanLN(STEM), nn.ReLU())
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.n0 = ChanLN(WIDTH)
        layers = [ResBlock(WIDTH)]
        if TOKEN_MIX:
            layers.append(TokenMix(WIDTH, cells))
        layers.append(ResBlock(WIDTH))
        if TOKEN_MIX:
            layers.append(TokenMix(WIDTH, cells))
        self.layers = nn.ModuleList(layers)
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        if self.grid:
            self.out = nn.Conv2d(HEAD, self.K, 1)
        else:
            self.out = nn.Conv2d(HEAD, MASK_EMB, 1)
            self.place_lin = nn.Linear(MASK_EMB * cells, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        outs = [self.out, self.done_head, self.value_head]
        if DENSE_MLP:
            self.mlp_in = nn.Linear(self.C * cells + self.n_actions, MLP_HID)
            self.mlp_norm = nn.LayerNorm(MLP_HID)
            self.mlp_out = nn.Linear(MLP_HID, self.n_actions)
            self.mlp_v = nn.Linear(MLP_HID, 1)
            outs += [self.mlp_out, self.mlp_v]
        for m in outs:
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.zeros_(m.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K = self.Y, self.X, self.K
        bases = [obs]
        if REINTERP:
            bases.append(obs.reshape(B, Y, X, self.C).permute(0, 3, 1, 2))
        views = torch.cat([sym(o, *s) for o in bases for s in self.syms], dim=1)
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        mfull = mask.to(obs.dtype)
        if DONE_FIRST:
            mp = mfull[:, 1:1 + self.n_place]
        else:
            mp = mfull[:, :self.n_place]
        if self.grid:
            mi = mp.reshape(B, K, Y, X)
        else:
            mi = self.mask_embed(mp).reshape(B, MASK_EMB, Y, X)
        s = self.obs_stem(torch.cat([views, ys, xs], dim=1)) + self.mask_stem(mi)
        if INST_NORM:
            z = F.instance_norm(self.dw(views))
            s = s + self.inst_stem(z)
        x = F.relu(self.n0(self.stem2(s)))
        for layer in self.layers:
            x = layer(x)
        h = F.relu(self.h3(x) + self.h1(x))
        if self.grid:
            place = self.out(h).reshape(B, -1)
        else:
            place = self.place_lin(self.out(h).flatten(1))
        g = self.glob(pooled(x))
        done = self.done_head(g)
        value = self.value_head(g).squeeze(1)
        if DONE_FIRST:
            logits = torch.cat([done, place], dim=1)
        else:
            logits = torch.cat([place, done], dim=1)
        if DENSE_MLP:
            hid = F.relu(self.mlp_norm(self.mlp_in(torch.cat([obs.reshape(B, -1), mfull], dim=1))))
            logits = logits + self.mlp_out(hid)
            value = value + self.mlp_v(hid).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
