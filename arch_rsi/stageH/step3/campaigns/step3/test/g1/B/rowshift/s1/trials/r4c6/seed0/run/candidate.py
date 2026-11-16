import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 1, "ent_coef": 0.003, "gamma": 0.9, "gae_lambda": 0.85}

STEM = 64
WIDTH = 40
HEAD = 32
WIDE = True
OFFSETS = (0, 1)
ATTN = True
HEADS = 4
ZFACT = True
ZW = 12
MEMB = 8


def layouts():
    if WIDE:
        res = []
        for tm in (0, 1):
            for tr in (0, 1):
                for fy in (0, 1):
                    for fx in (0, 1):
                        res.append((tr, fy, fx, tm))
        return res
    return [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)]


def layout_index(T, Y, X, tr, fy, fx, tm):
    idx = []
    for t in range(T):
        for y in range(Y):
            for x in range(X):
                yy = Y - 1 - y if fy else y
                xx = X - 1 - x if fx else x
                cell = xx * Y + yy if tr else yy * X + xx
                idx.append(cell * T + t if tm else t * Y * X + cell)
    return idx


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y, X, T, Z = int(spec.Y), int(spec.X), int(spec.n_types), int(spec.Z)
        C = int(spec.in_channels)
        self.Y, self.X, self.T, self.Z = Y, X, T, Z
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.ok = (self.n_place == T * Y * X) and (self.n_actions == self.n_place + 1)
        self.zok = ZFACT and (C == 4 * Z + 3)
        cin = C + 2
        if self.zok:
            self.zc1 = nn.Conv2d(5, ZW, 3, padding=1)
            self.zc2 = nn.Conv2d(ZW, ZW, 3, padding=1)
            cin += 2 * ZW
        if self.ok:
            gi, si, offs = [], [], []
            for o in OFFSETS:
                for lay in layouts():
                    idx = layout_index(T, Y, X, lay[0], lay[1], lay[2], lay[3])
                    inv = [0] * len(idx)
                    for p, a in enumerate(idx):
                        inv[a] = p
                    gi.append([a + o for a in idx])
                    si.append(inv)
                    offs.append(o)
            self.register_buffer("gidx", torch.tensor(gi, dtype=torch.long).flatten())
            self.register_buffer("sidx", torch.tensor(si, dtype=torch.long))
            self.offs = offs
            self.L = len(offs)
            cin += self.L * T
            nout = self.L * T
        else:
            self.offs = []
            self.L = 0
            self.membed = nn.Linear(self.n_actions, MEMB * Y * X)
            cin += MEMB
            nout = MEMB
            self.fallback = nn.Linear(MEMB * Y * X, self.n_actions - 1)
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(),
                                  nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        if ATTN:
            self.ln = nn.LayerNorm(WIDTH)
            self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
            self.aout = nn.Linear(WIDTH, WIDTH)
        self.hconv = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.hskip = nn.Conv2d(WIDTH, HEAD, 1)
        self.hout = nn.Conv2d(HEAD, nout, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T, Z = self.Y, self.X, self.T, self.Z
        m = mask.to(obs.dtype)
        if self.ok:
            mi = m[:, self.gidx].reshape(B, self.L * T, Y, X)
        else:
            mi = self.membed(m).reshape(B, MEMB, Y, X)
        cy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        cx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        parts = [obs, mi, cy, cx]
        if self.zok:
            vol = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).transpose(1, 2)
            zc = torch.linspace(-1.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
            v = torch.cat([vol, zc], 2).reshape(B * Z, 5, Y, X)
            v = F.relu(self.zc1(v))
            v = F.relu(v + self.zc2(v))
            v = v.reshape(B, Z, ZW, Y, X)
            parts.append(v.amax(1))
            parts.append(v.mean(1))
        h = self.stem(torch.cat(parts, 1))
        h = F.relu(h + self.r1(h))
        f = F.relu(h + self.r2(h))
        if ATTN:
            N = Y * X
            dh = WIDTH // HEADS
            t = self.ln(f.flatten(2).transpose(1, 2))
            q, k, vv = self.qkv(t).chunk(3, -1)
            q = q.reshape(B, N, HEADS, dh).transpose(1, 2)
            k = k.reshape(B, N, HEADS, dh).transpose(1, 2)
            vv = vv.reshape(B, N, HEADS, dh).transpose(1, 2)
            att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh), -1)
            ao = torch.matmul(att, vv).transpose(1, 2).reshape(B, N, WIDTH)
            f = F.relu(f + self.aout(ao).transpose(1, 2).reshape(B, WIDTH, Y, X))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        done = self.done_head(g)
        hd = F.relu(self.hconv(f) + self.hskip(f))
        out = self.hout(hd)
        if self.ok:
            out = out.reshape(B, self.L, T * Y * X)
            logits = F.pad(done, (self.n_place, 0))
            if 1 in self.offs:
                logits = logits + F.pad(done, (0, self.n_place))
            for i in range(self.L):
                o = self.offs[i]
                part = out[:, i].index_select(1, self.sidx[i])
                logits = logits + F.pad(part, (o, self.n_actions - self.n_place - o))
        else:
            logits = torch.cat([self.fallback(out.flatten(1)), done], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
