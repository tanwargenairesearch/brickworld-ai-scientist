import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.85}
LAYOUTS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
TYPE_MINOR = False
OBS_SYM = True
USE_LN = True
GLOB_CTX = True
DILATED = True
DENSE_REMAP = False
SEP_CRITIC = True
STEM = 64
WIDTH = 40
HEAD = 32
GLOB = 64
MASK_EMB = 8
DENSE_EMB = 4
DENSE_OUT = 4
CRIT = 32


def lay(t, tr, fy, fx):
    if tr:
        t = t.transpose(-1, -2)
    if fy:
        t = t.flip(-2)
    if fx:
        t = t.flip(-1)
    return t


def unlay(t, tr, fy, fx):
    if fx:
        t = t.flip(-1)
    if fy:
        t = t.flip(-2)
    if tr:
        t = t.transpose(-1, -2)
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


class ResBlock(nn.Module):
    def __init__(self, c, dil):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=dil, dilation=dil)
        self.norm = norm(c)
        self.use_ctx = GLOB_CTX
        if GLOB_CTX:
            self.ctx = nn.Linear(2 * c, c)

    def forward(self, x):
        y = x + self.conv(x)
        if self.use_ctx:
            g = self.ctx(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
            y = y + g[:, :, None, None]
        return F.relu(self.norm(y))


class Critic(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.c1 = nn.Conv2d(cin, CRIT, 1)
        self.n1 = ChanLN(CRIT)
        self.c2 = nn.Conv2d(CRIT, CRIT, 3, stride=2, padding=1)
        self.fc1 = nn.Linear(2 * CRIT, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, o):
        h = F.relu(self.n1(self.c1(o)))
        h = F.relu(self.c2(h))
        h = F.relu(self.fc1(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)))
        return self.fc2(h).squeeze(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        cells = self.Y * self.X
        square = self.Y == self.X
        lays = [l for l in LAYOUTS if square or not l[0]]
        if len(lays) == 0:
            lays = [(0, 0, 0)]
        self.layouts = lays
        self.grid = (self.n_place % cells == 0)
        if OBS_SYM:
            syms = [(t, a, b) for t in (0, 1) for a in (0, 1) for b in (0, 1)]
            self.syms = [s for s in syms if square or not s[0]]
        else:
            self.syms = [(0, 0, 0)]
        obs_in = int(spec.in_channels) * len(self.syms) + 2
        if self.grid:
            self.K = self.n_place // cells
            mask_in = self.K * len(self.layouts)
        else:
            self.K = 1
            self.mask_embed = nn.Linear(self.n_place, cells * MASK_EMB)
            mask_in = MASK_EMB
        if DENSE_REMAP:
            self.dense_in = nn.Linear(self.n_actions, cells * DENSE_EMB)
            mask_in += DENSE_EMB
        if USE_LN:
            self.obs_stem = nn.Sequential(nn.Conv2d(obs_in, STEM, 1), ChanLN(STEM), nn.ReLU())
            self.mask_stem = nn.Sequential(nn.Conv2d(mask_in, STEM, 1), ChanLN(STEM), nn.ReLU())
        else:
            self.stem1 = nn.Conv2d(obs_in + mask_in, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.n0 = norm(WIDTH)
        blocks = [ResBlock(WIDTH, 1)]
        if DILATED:
            blocks.append(ResBlock(WIDTH, 2))
        blocks.append(ResBlock(WIDTH, 1))
        self.blocks = nn.ModuleList(blocks)
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        if self.grid:
            self.out = nn.Conv2d(HEAD, self.K * len(self.layouts), 1)
        else:
            self.out = nn.Conv2d(HEAD, MASK_EMB, 1)
            self.place_lin = nn.Linear(MASK_EMB * cells, self.n_place)
        if DENSE_REMAP:
            self.dense_feat = nn.Conv2d(HEAD, DENSE_OUT, 1)
            self.dense_out = nn.Linear(DENSE_OUT * cells + GLOB, self.n_actions)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        if SEP_CRITIC:
            self.critic = Critic(int(spec.in_channels))
        if USE_LN:
            outs = [self.out, self.done_head, self.value_head]
            if DENSE_REMAP:
                outs.append(self.dense_out)
            if SEP_CRITIC:
                outs.append(self.critic.fc2)
            for m in outs:
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        K = self.K
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        o = torch.cat([lay(obs, *s) for s in self.syms] + [ys, xs], dim=1)
        mfull = mask.to(obs.dtype)
        mp = mfull[:, :self.n_place]
        if self.grid:
            if TYPE_MINOR:
                m4 = mp.reshape(B, Y, X, K).permute(0, 3, 1, 2)
            else:
                m4 = mp.reshape(B, K, Y, X)
            mi = torch.cat([lay(m4, *l) for l in self.layouts], dim=1)
        else:
            mi = self.mask_embed(mp).reshape(B, MASK_EMB, Y, X)
        if DENSE_REMAP:
            mi = torch.cat([mi, self.dense_in(mfull).reshape(B, DENSE_EMB, Y, X)], dim=1)
        if USE_LN:
            s = self.obs_stem(o) + self.mask_stem(mi)
        else:
            s = F.relu(self.stem1(torch.cat([o, mi], dim=1)))
        x = F.relu(self.n0(self.stem2(s)))
        for blk in self.blocks:
            x = blk(x)
        h = F.relu(self.h3(x) + self.h1(x))
        out = self.out(h)
        if self.grid:
            place = None
            for i, l in enumerate(self.layouts):
                part = unlay(out[:, i * K:(i + 1) * K], *l)
                if TYPE_MINOR:
                    part = part.permute(0, 2, 3, 1)
                part = part.reshape(B, -1)
                place = part if place is None else place + part
        else:
            place = self.place_lin(out.flatten(1))
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        if DENSE_REMAP:
            logits = logits + self.dense_out(torch.cat([self.dense_feat(h).flatten(1), g], dim=1))
        if SEP_CRITIC:
            value = self.critic(obs)
        else:
            value = self.value_head(g).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
