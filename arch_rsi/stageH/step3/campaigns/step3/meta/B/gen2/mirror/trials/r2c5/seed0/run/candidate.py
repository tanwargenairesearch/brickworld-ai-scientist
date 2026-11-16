import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

LAYOUTS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
USE_LN = True
OUT_GAIN = 0.0
GCTX = False
STEM = 64
WIDTH = 40
HEAD = 32
HID = 64


def lay(v, t, fy, fx):
    if t:
        v = v.transpose(2, 3)
    if fy:
        v = v.flip(2)
    if fx:
        v = v.flip(3)
    return v


def unlay(g, t, fy, fx):
    if fx:
        g = g.flip(3)
    if fy:
        g = g.flip(2)
    if t:
        g = g.transpose(2, 3)
    return g


class CellLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def norm(c):
    return CellLN(c) if USE_LN else nn.Identity()


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        self.Y = Y
        self.X = X
        self.n_place = int(spec.n_place)
        sq = Y == X
        lays = [tuple(l) for l in LAYOUTS if sq or l[0] == 0]
        if len(lays) == 0:
            lays = [(0, 0, 0)]
        self.lays = lays
        self.grid = (self.n_place % (Y * X)) == 0
        self.K = self.n_place // (Y * X) if self.grid else 4
        c_obs = int(spec.in_channels) + 2
        if self.grid:
            c_mask = self.K * len(lays)
            n_out = self.K * len(lays)
        else:
            self.mask_embed = nn.Linear(self.n_place, 4 * Y * X)
            self.place_fc = nn.Linear(4 * Y * X, self.n_place)
            c_mask = 4
            n_out = 4
        self.stem_o = nn.Conv2d(c_obs, STEM, 1)
        self.stem_m = nn.Conv2d(c_mask, STEM, 1)
        self.ln_o = norm(STEM)
        self.ln_m = norm(STEM)
        self.proj = nn.Conv2d(STEM, WIDTH, 1)
        self.ln_p = norm(WIDTH)
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(2)])
        self.res_ln = nn.ModuleList([norm(WIDTH) for _ in range(2)])
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        if GCTX:
            self.gctx = nn.Linear(HID, HEAD)
        self.out = nn.Conv2d(HEAD, n_out, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, HID), nn.ReLU())
        self.done_head = nn.Linear(HID, 1)
        self.value_head = nn.Linear(HID, 1)
        if OUT_GAIN > 0:
            for m in (self.out, self.done_head):
                nn.init.orthogonal_(m.weight, gain=OUT_GAIN)
                nn.init.zeros_(m.bias)
            nn.init.orthogonal_(self.value_head.weight, gain=1.0)
            nn.init.zeros_(self.value_head.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = self.Y
        X = self.X
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        o_in = torch.cat([obs, yy, xx], dim=1)
        mp = mask[:, :self.n_place].to(obs.dtype)
        if self.grid:
            mv = mp.reshape(B, self.K, Y, X)
            m_in = torch.cat([lay(mv, t, a, b) for (t, a, b) in self.lays], dim=1)
        else:
            m_in = self.mask_embed(mp).reshape(B, 4, Y, X)
        o = self.stem_o(o_in)
        m = self.stem_m(m_in)
        if USE_LN:
            s = F.relu(self.ln_o(o)) + F.relu(self.ln_m(m))
        else:
            s = F.relu(o + m)
        x = F.relu(self.ln_p(self.proj(s)))
        for conv, ln in zip(self.res, self.res_ln):
            x = F.relu(x + ln(conv(x)))
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        hp = self.h3(x) + self.h1(x)
        if GCTX:
            hp = hp + self.gctx(g)[:, :, None, None]
        h = F.relu(hp)
        out = self.out(h)
        if self.grid:
            K = self.K
            place = 0
            for i, (t, a, b) in enumerate(self.lays):
                gg = unlay(out[:, i * K:(i + 1) * K], t, a, b)
                place = place + gg.reshape(B, -1)
        else:
            place = self.place_fc(out.reshape(B, -1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
