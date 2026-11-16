import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

ATTN_BLOCKS = 0
HEADS = 4
FLAT = True
ACT_ATTN = False
PAD_MODE = "zeros"
MAX_SCHEMES = 16
EMB = 8
STEM = 64
WIDTH = 40
HEADC = 32
GLOB = 64
FLATH = 128
AD = 32
COMPACT_KINDS = ["row", "col", "swap", "uni"]


def conv3(cin, cout):
    return nn.Conv2d(cin, cout, 3, padding=1, padding_mode=PAD_MODE)


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


class SelfAttn(nn.Module):
    def __init__(self, c, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(c, 3 * c)
        self.proj = nn.Linear(c, c)
        self.ff1 = nn.Linear(c, 2 * c)
        self.ff2 = nn.Linear(2 * c, c)
        self.n1 = nn.LayerNorm(c)
        self.n2 = nn.LayerNorm(c)

    def forward(self, f):
        B, C, Y, X = f.shape
        t = f.flatten(2).transpose(1, 2)
        q, k, v = self.qkv(self.n1(t)).chunk(3, dim=-1)
        H = self.h
        d = C // H
        q = q.reshape(B, -1, H, d).transpose(1, 2)
        k = k.reshape(B, -1, H, d).transpose(1, 2)
        v = v.reshape(B, -1, H, d).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d), dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, -1, C)
        t = t + self.proj(o)
        t = t + self.ff2(F.relu(self.ff1(self.n2(t))))
        return t.transpose(1, 2).reshape(B, C, Y, X)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        T = int(spec.n_types)
        C = int(spec.in_channels)
        self.Y, self.X, self.C = Y, X, C
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
        in_ch = C + self.n_maps * G + 2
        if self.use_dense:
            self.mask_embed = nn.Linear(self.n_actions, EMB * Y * X)
            in_ch = in_ch + EMB
        self.stem = nn.Sequential(nn.Conv2d(in_ch, STEM, 1), nn.ReLU(),
                                  nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = conv3(WIDTH, WIDTH)
        self.r2 = conv3(WIDTH, WIDTH)
        self.attn = nn.ModuleList([SelfAttn(WIDTH, HEADS) for _ in range(ATTN_BLOCKS)])
        self.hconv = conv3(WIDTH, HEADC)
        self.hskip = nn.Conv2d(WIDTH, HEADC, 1)
        if self.n_maps > 0:
            self.hout = nn.Conv2d(HEADC, self.n_maps * G, 1)
            self.scale = min(1.0, 2.0 / (self.n_maps ** 0.5))
        if self.use_dense:
            self.dense_red = nn.Conv2d(HEADC, 8, 1)
            self.dense_out = nn.Linear(8 * Y * X, self.n_actions)
        gin = 2 * WIDTH
        if FLAT:
            self.flat_obs = nn.Linear(C * Y * X, FLATH)
            self.flat_mask = nn.Linear(self.n_actions, FLATH)
            self.flat_mix = nn.Linear(FLATH, FLATH)
            self.flat_bc = nn.Linear(FLATH, WIDTH)
            self.flat_out = nn.Linear(FLATH, self.n_actions)
            gin = gin + FLATH
        self.glob = nn.Sequential(nn.Linear(gin, GLOB), nn.ReLU())
        self.bias_head = nn.Linear(GLOB, self.n_actions)
        self.value_head = nn.Linear(GLOB, 1)
        if ACT_ATTN:
            self.aemb = nn.Embedding(self.n_actions, AD)
            self.amask = nn.Linear(1, AD)
            self.aglob = nn.Linear(GLOB, AD)
            self.akey = nn.Conv2d(HEADC, AD, 1)
            self.aval = nn.Conv2d(HEADC, AD, 1)
            self.aout1 = nn.Linear(AD, AD)
            self.aout2 = nn.Linear(AD, 1)
            self.register_buffer("aidx", torch.arange(self.n_actions, dtype=torch.long))

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        G = self.G
        m = mask.float()
        parts = [obs]
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
        if FLAT:
            z = F.relu(self.flat_obs(obs.flatten(1)) + self.flat_mask(m))
            z = F.relu(self.flat_mix(z))
            h = h + self.flat_bc(z).view(B, -1, 1, 1)
        h = F.relu(h + self.r1(h))
        f = F.relu(h + self.r2(h))
        for blk in self.attn:
            f = blk(f)
        hh = F.relu(self.hconv(f) + self.hskip(f))
        pooled = torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)
        if FLAT:
            pooled = torch.cat([pooled, z], 1)
        gf = self.glob(pooled)
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
        if FLAT:
            logits = logits + self.flat_out(z)
        if ACT_ATTN:
            q = self.aemb(self.aidx).unsqueeze(0) + self.amask(m.unsqueeze(-1)) + self.aglob(gf).unsqueeze(1)
            kk = self.akey(hh).flatten(2)
            vv = self.aval(hh).flatten(2).transpose(1, 2)
            att = torch.softmax(torch.matmul(q, kk) / math.sqrt(AD), dim=-1)
            oo = torch.matmul(att, vv)
            logits = logits + self.aout2(F.relu(self.aout1(q + oo))).squeeze(-1)
        return logits, self.value_head(gf).squeeze(1)


def build_policy(spec):
    return Net(spec)
