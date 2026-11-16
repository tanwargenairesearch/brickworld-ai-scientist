import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

OFFSETS = [0, 1]
# (type_major, spatial_transposed, flip_y, flip_x)
FULL_LAYOUTS = [(True, False, False, False), (True, False, False, True),
                (True, False, True, False), (True, False, True, True),
                (True, True, False, False), (True, True, False, True),
                (True, True, True, False), (True, True, True, True),
                (False, False, False, False), (False, False, False, True),
                (False, False, True, False), (False, False, True, True),
                (False, True, False, False), (False, True, False, True),
                (False, True, True, False), (False, True, True, True)]
COMPACT_KINDS = ["row", "col", "swap", "uni"]
OBS_VIEWS = ["id", "chl"]
DENSE = True
EMB = 8
GLOBAL_CTX = False
EXTRA_BLOCKS = 0
HEAD_K = 5
MAX_SCHEMES = 16
STEM = 64
WIDTH = 40
HEADC = 32
GLOB = 64


def full_scheme(T, Y, X, lay):
    tmaj, tr, fy, fx = lay
    out = []
    for a in range(T * Y * X):
        if tmaj:
            t = a // (Y * X)
            r = a % (Y * X)
        else:
            t = a % T
            r = a // T
        if tr:
            x = r // Y
            y = r % Y
        else:
            y = r // X
            x = r % X
        if fy:
            y = Y - 1 - y
        if fx:
            x = X - 1 - x
        out.append(t * Y * X + y * X + x)
    return out


def compact_scheme(T, Y, X, fps, kind):
    mdy = max([f[0] for f in fps] + [1])
    mdx = max([f[1] for f in fps] + [1])
    out = []
    for t in range(T):
        if t < len(fps):
            dy, dx = fps[t]
        else:
            dy, dx = 1, 1
        if kind == "swap":
            dy, dx = dx, dy
        if kind == "uni":
            dy, dx = mdy, mdx
        ny = max(Y - dy + 1, 0)
        nx = max(X - dx + 1, 0)
        if kind == "col":
            for x in range(nx):
                for y in range(ny):
                    out.append(t * Y * X + y * X + x)
        else:
            for y in range(ny):
                for x in range(nx):
                    out.append(t * Y * X + y * X + x)
    return out


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        T, Y, X = self.T, self.Y, self.X
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        cands = [full_scheme(T, Y, X, lay) for lay in FULL_LAYOUTS]
        cands = cands + [compact_scheme(T, Y, X, fps, k) for k in COMPACT_KINDS]
        schemes = []
        for s in cands:
            if len(s) == self.n_place and len(s) > 0 and s not in schemes:
                schemes.append(s)
        schemes = schemes[:MAX_SCHEMES]
        self.offsets = [o for o in OFFSETS if o + self.n_place <= self.n_actions]
        if len(schemes) == 0 or len(self.offsets) == 0:
            schemes = []
            self.offsets = []
        self.n_s = len(schemes)
        if self.n_s > 0:
            self.register_buffer("idx", torch.tensor(schemes, dtype=torch.long))
        else:
            self.register_buffer("idx", torch.zeros(1, 1, dtype=torch.long))
        self.n_maps = self.n_s * len(self.offsets)
        self.use_dense = DENSE or self.n_maps == 0
        self.views = [v for v in OBS_VIEWS if v != "tr" or Y == X]
        in_ch = len(self.views) * self.C + self.n_maps * T + 2
        if self.use_dense:
            self.mask_embed = nn.Linear(self.n_actions, EMB * Y * X)
            in_ch = in_ch + EMB
        self.stem = nn.Sequential(nn.Conv2d(in_ch, STEM, 1), nn.ReLU(),
                                  nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.extra = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=2, dilation=2) for _ in range(EXTRA_BLOCKS)])
        if GLOBAL_CTX:
            self.ctx = nn.Linear(2 * WIDTH, WIDTH)
        if HEAD_K == 3:
            self.hpre = nn.Identity()
            self.hconv = nn.Conv2d(WIDTH, HEADC, 3, padding=1)
        else:
            self.hpre = nn.Conv2d(WIDTH, 16, 1)
            self.hconv = nn.Conv2d(16, HEADC, HEAD_K, padding=HEAD_K // 2)
        self.hskip = nn.Conv2d(WIDTH, HEADC, 1)
        if self.n_maps > 0:
            self.hout = nn.Conv2d(HEADC, self.n_maps * T, 1)
            self.scale = min(1.0, 2.0 / (self.n_maps ** 0.5))
        if self.use_dense:
            self.dense_red = nn.Conv2d(HEADC, 8, 1)
            self.dense_out = nn.Linear(8 * Y * X, self.n_actions)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.bias_head = nn.Linear(GLOB, self.n_actions)
        self.value_head = nn.Linear(GLOB, 1)

    def obs_views(self, obs):
        B = obs.shape[0]
        out = []
        for v in self.views:
            if v == "id":
                out.append(obs)
            elif v == "tr":
                out.append(obs.transpose(2, 3))
            elif v == "chl":
                out.append(obs.reshape(B, self.Y, self.X, self.C).permute(0, 3, 1, 2))
        return out

    def forward(self, obs, mask):
        B = obs.shape[0]
        T, Y, X = self.T, self.Y, self.X
        m = mask.float()
        parts = self.obs_views(obs)
        for o in self.offsets:
            mp = m[:, o:o + self.n_place]
            for k in range(self.n_s):
                g = m.new_zeros(B, T * Y * X).index_copy(1, self.idx[k], mp)
                parts.append(g.view(B, T, Y, X))
        if self.use_dense:
            parts.append(self.mask_embed(m).view(B, EMB, Y, X))
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        parts.append(ys)
        parts.append(xs)
        h = self.stem(torch.cat(parts, 1))
        h = F.relu(h + self.r1(h))
        f = F.relu(h + self.r2(h))
        if GLOBAL_CTX:
            c = self.ctx(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
            f = F.relu(f + c.view(B, WIDTH, 1, 1))
        for blk in self.extra:
            f = F.relu(f + blk(f))
        hh = F.relu(self.hconv(self.hpre(f)) + self.hskip(f))
        gf = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        logits = self.bias_head(gf)
        if self.n_maps > 0:
            maps = self.hout(hh)
            j = 0
            for o in self.offsets:
                acc = 0
                for k in range(self.n_s):
                    acc = acc + maps[:, j * T:(j + 1) * T].flatten(1).index_select(1, self.idx[k])
                    j += 1
                logits = logits + F.pad(acc * self.scale, (o, self.n_actions - self.n_place - o))
        if self.use_dense:
            logits = logits + self.dense_out(self.dense_red(hh).flatten(1))
        return logits, self.value_head(gf).squeeze(1)


def build_policy(spec):
    return Net(spec)
