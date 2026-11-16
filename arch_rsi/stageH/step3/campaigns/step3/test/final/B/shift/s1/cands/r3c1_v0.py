import torch
import torch.nn as nn
import torch.nn.functional as F

LAYOUTS = ()
TYPE_MINOR = False
DENSE = True
SEP_CRITIC = False
STEM = 64
WIDTH = 40
HEAD = 32
GH = 64
DPROJ = 16
MEMB = 4
CW = 32
RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}


def fwd_layout(t, lay):
    tr, fy, fx = lay
    if tr:
        t = t.transpose(-1, -2)
    if fy:
        t = t.flip(-2)
    if fx:
        t = t.flip(-1)
    return t


def inv_layout(t, lay):
    tr, fy, fx = lay
    if fx:
        t = t.flip(-1)
    if fy:
        t = t.flip(-2)
    if tr:
        t = t.transpose(-1, -2)
    return t


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        self.Y = Y
        self.X = X
        self.n_place = int(spec.n_place)
        self.n_act = int(spec.n_actions)
        hw = Y * X
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.fps = fps
        nt = len(fps)
        compact = [max(Y - dy + 1, 0) * max(X - dx + 1, 0) for dy, dx in fps]
        if nt * hw == self.n_place:
            self.mode = 0
            self.K = nt
        elif sum(compact) == self.n_place and min(compact) > 0:
            self.mode = 1
            self.K = nt
        else:
            self.mode = 2
            self.K = (self.n_place + hw - 1) // hw
        self.lays = [l for l in LAYOUTS if Y == X or l[0] == 0]
        c_obs = spec.in_channels + 2
        c_mask = self.K * len(self.lays)
        self.memb = nn.Linear(self.n_act, MEMB * hw)
        self.stem = nn.Sequential(nn.Conv2d(c_obs + MEMB + c_mask, STEM, 1), nn.ReLU(),
                                  nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        if len(self.lays) > 0:
            self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
            self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
            self.out = nn.Conv2d(HEAD, self.K * len(self.lays), 1)
            self.done_head = nn.Linear(GH, 1)
        if DENSE or len(self.lays) == 0:
            self.dproj = nn.Conv2d(WIDTH, DPROJ, 1)
            self.dense = nn.Linear(DPROJ * hw + GH, self.n_act)
        if SEP_CRITIC:
            self.cstem = nn.Conv2d(c_obs, CW, 1)
            self.cconv = nn.Conv2d(CW, CW, 3, padding=1)
            self.cfc = nn.Linear(2 * CW, GH)
            self.cval = nn.Linear(GH, 1)
        else:
            self.value_head = nn.Linear(GH, 1)

    def mask_grid(self, mp):
        B = mp.shape[0]
        Y, X, K = self.Y, self.X, self.K
        if self.mode == 1:
            parts = []
            off = 0
            for dy, dx in self.fps:
                h = Y - dy + 1
                w = X - dx + 1
                blk = mp[:, off:off + h * w].reshape(B, 1, h, w)
                parts.append(F.pad(blk, (0, X - w, 0, Y - h)))
                off += h * w
            return torch.cat(parts, 1)
        pad = K * Y * X - self.n_place
        if pad > 0:
            mp = F.pad(mp, (0, pad))
        if TYPE_MINOR:
            return mp.reshape(B, Y, X, K).permute(0, 3, 1, 2)
        return mp.reshape(B, K, Y, X)

    def grid_logits(self, g):
        B = g.shape[0]
        Y, X = self.Y, self.X
        if self.mode == 1:
            parts = []
            for i, (dy, dx) in enumerate(self.fps):
                parts.append(g[:, i, :Y - dy + 1, :X - dx + 1].reshape(B, -1))
            return torch.cat(parts, 1)
        if TYPE_MINOR:
            g = g.permute(0, 2, 3, 1)
        return g.reshape(B, -1)[:, :self.n_place]

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K = self.Y, self.X, self.K
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        o = torch.cat([obs, ys, xs], 1)
        mf = mask.to(obs.dtype)
        me = self.memb(mf).reshape(B, MEMB, Y, X)
        parts = [o, me]
        if len(self.lays) > 0:
            mg = self.mask_grid(mf[:, :self.n_place])
            parts = parts + [fwd_layout(mg, l) for l in self.lays]
        x = self.stem(torch.cat(parts, 1))
        x = F.relu(x + self.r1(x))
        x = F.relu(x + self.r2(x))
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))
        logits = None
        if len(self.lays) > 0:
            h = F.relu(self.h3(x) + self.h1(x))
            maps = self.out(h)
            place = 0
            for i, l in enumerate(self.lays):
                place = place + self.grid_logits(inv_layout(maps[:, i * K:(i + 1) * K], l))
            logits = torch.cat([place, self.done_head(g)], 1)
            extra = self.n_act - logits.shape[1]
            if extra > 0:
                logits = F.pad(logits, (0, extra))
        if DENSE or len(self.lays) == 0:
            d = F.relu(self.dproj(x)).flatten(1)
            dl = self.dense(torch.cat([d, g], 1))
            logits = dl if logits is None else logits + dl
        if SEP_CRITIC:
            c = F.relu(self.cstem(o))
            c = F.relu(c + self.cconv(c))
            cv = F.relu(self.cfc(torch.cat([c.mean((2, 3)), c.amax((2, 3))], 1)))
            value = self.cval(cv).squeeze(1)
        else:
            value = self.value_head(g).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
