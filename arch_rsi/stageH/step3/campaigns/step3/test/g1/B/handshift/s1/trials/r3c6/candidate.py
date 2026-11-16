import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 1.5e-3, "n_envs": 256, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.93, "gae_lambda": 0.9}

OBS_VIEWS = ["id"]
NORM_IN = True
LAYERWISE = False
LW = 12
HEAD_K = 3
MAX_SCHEMES = 16
EMB = 8
STEM = 64
WIDTH = 40
HEADC = 32
GLOB = 64
COMPACT_KINDS = ["row", "col", "swap", "uni"]


def perm_scheme(K, Y, X, tmaj, tr, fy, fx):
    out = []
    for a in range(K * Y * X):
        if tmaj:
            k = a // (Y * X)
            r = a % (Y * X)
        else:
            k = a % K
            r = a // K
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
        out.append(k * Y * X + y * X + x)
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
        Y = int(spec.Y)
        X = int(spec.X)
        Z = int(spec.Z)
        T = int(spec.n_types)
        C = int(spec.in_channels)
        self.Y, self.X, self.Z, self.C = Y, X, Z, C
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        schemes = []
        G = T
        if self.n_place > 0 and self.n_place % (Y * X) == 0:
            G = self.n_place // (Y * X)
            for tmaj in (True, False):
                for tr in (False, True):
                    for fy in (False, True):
                        for fx in (False, True):
                            s = perm_scheme(G, Y, X, tmaj, tr, fy, fx)
                            if s not in schemes:
                                schemes.append(s)
        else:
            for kind in COMPACT_KINDS:
                s = compact_scheme(T, Y, X, fps, kind)
                if len(s) == self.n_place and len(s) > 0 and s not in schemes:
                    schemes.append(s)
        schemes = schemes[:MAX_SCHEMES]
        self.G = G
        offs = []
        for o in [0, self.n_actions - self.n_place]:
            if o >= 0 and o + self.n_place <= self.n_actions and o not in offs:
                offs.append(o)
        if len(schemes) == 0 or len(offs) == 0:
            schemes = []
            offs = []
        self.offsets = offs
        self.n_s = len(schemes)
        if self.n_s > 0:
            self.register_buffer("idx", torch.tensor(schemes, dtype=torch.long))
        else:
            self.register_buffer("idx", torch.zeros(1, 1, dtype=torch.long))
        self.n_maps = self.n_s * len(self.offsets)
        self.use_dense = self.n_maps == 0
        self.views = [v for v in OBS_VIEWS if v != "tr" or Y == X]
        nv = len(self.views) * C
        if NORM_IN:
            self.inorm = nn.Conv2d(nv, nv, 1, groups=nv)
        self.use_lw = LAYERWISE and C == 4 * Z + 3 and Z > 0
        in_ch = nv + self.n_maps * G + 2
        if self.use_lw:
            self.lw1 = nn.Conv2d(12, LW, 3, padding=1)
            self.lw2 = nn.Conv2d(LW, LW, 3, padding=1)
            in_ch = in_ch + Z * LW
        if self.use_dense:
            self.mask_embed = nn.Linear(self.n_actions, EMB * Y * X)
            in_ch = in_ch + EMB
        self.stem = nn.Sequential(nn.Conv2d(in_ch, STEM, 1), nn.ReLU(),
                                  nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        if NORM_IN:
            self.n1 = nn.GroupNorm(8, WIDTH)
            self.n2 = nn.GroupNorm(8, WIDTH)
        else:
            self.n1 = nn.Identity()
            self.n2 = nn.Identity()
        if HEAD_K == 3:
            self.hpre = nn.Identity()
            self.hconv = nn.Conv2d(WIDTH, HEADC, 3, padding=1)
        else:
            self.hpre = nn.Conv2d(WIDTH, 16, 1)
            self.hconv = nn.Conv2d(16, HEADC, HEAD_K, padding=HEAD_K // 2)
        self.hskip = nn.Conv2d(WIDTH, HEADC, 1)
        if self.n_maps > 0:
            self.hout = nn.Conv2d(HEADC, self.n_maps * G, 1)
            self.scale = min(1.0, 2.0 / (self.n_maps ** 0.5))
        if self.use_dense:
            self.dense_red = nn.Conv2d(HEADC, 8, 1)
            self.dense_out = nn.Linear(8 * Y * X, self.n_actions)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.bias_head = nn.Linear(GLOB, self.n_actions)
        self.value_head = nn.Linear(GLOB, 1)

    def obs_views(self, obs):
        B, C, Y, X = obs.shape
        out = []
        for v in self.views:
            if v == "id":
                out.append(obs)
            elif v == "tr":
                out.append(obs.transpose(2, 3))
            elif v == "chl":
                out.append(obs.reshape(B, Y, X, C).permute(0, 3, 1, 2))
            elif v == "chlT":
                out.append(obs.reshape(B, X, Y, C).permute(0, 3, 2, 1))
        return out

    def layer_feats(self, obs):
        B, C, Y, X = obs.shape
        Z = self.Z
        v = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        below = F.pad(v, (0, 0, 0, 0, 1, 0))[:, :, :Z]
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = torch.linspace(-1.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        u = torch.cat([v, below, gl, zc], 1)
        u = u.permute(0, 2, 1, 3, 4).reshape(B * Z, 12, Y, X)
        u = F.relu(self.lw1(u))
        u = F.relu(self.lw2(u))
        return u.reshape(B, Z * LW, Y, X)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        G = self.G
        m = mask.float()
        vs = self.obs_views(obs)
        ov = torch.cat(vs, 1) if len(vs) > 1 else vs[0]
        if NORM_IN:
            ov = torch.tanh(self.inorm(ov))
        parts = [ov]
        if self.use_lw:
            parts.append(self.layer_feats(obs))
        for o in self.offsets:
            mp = m[:, o:o + self.n_place]
            for k in range(self.n_s):
                g = m.new_zeros(B, G * Y * X).index_copy(1, self.idx[k], mp)
                parts.append(g.view(B, G, Y, X))
        if self.use_dense:
            parts.append(self.mask_embed(m).view(B, EMB, Y, X))
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        parts.append(ys)
        parts.append(xs)
        h = self.stem(torch.cat(parts, 1))
        h = F.relu(h + self.n1(self.r1(h)))
        f = F.relu(h + self.n2(self.r2(h)))
        hh = F.relu(self.hconv(self.hpre(f)) + self.hskip(f))
        gf = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        logits = self.bias_head(gf)
        if self.n_maps > 0:
            maps = self.hout(hh)
            j = 0
            for o in self.offsets:
                acc = 0
                for k in range(self.n_s):
                    acc = acc + maps[:, j * G:(j + 1) * G].flatten(1).index_select(1, self.idx[k])
                    j += 1
                logits = logits + F.pad(acc * self.scale, (o, self.n_actions - self.n_place - o))
        if self.use_dense:
            logits = logits + self.dense_out(self.dense_red(hh).flatten(1))
        return logits, self.value_head(gf).squeeze(1)


def build_policy(spec):
    return Net(spec)
