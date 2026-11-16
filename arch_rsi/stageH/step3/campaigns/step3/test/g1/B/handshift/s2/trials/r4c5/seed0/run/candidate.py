import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 256, "rollout_len": 32, "n_minibatches": 16,
          "ppo_epochs": 1, "ent_coef": 0.03, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
WIDTH = 40
HEAD = 32
HEAD_K = 3
EMB = 8
# layout = (transpose, flip_y, flip_x)
LAYOUTS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
DONE_FIRST = False
OBS_SYM = True
CIRC = True
NORM = False
GLOBAL_CTX = False
C3D = 0


def pad_mode():
    return "circular" if CIRC else "zeros"


def gn(c):
    return nn.GroupNorm(8, c) if NORM else nn.Identity()


def to_grid(m, lay, K, Y, X):
    tr, fy, fx = lay
    B = m.shape[0]
    if tr:
        g = m.reshape(B, K, X, Y).transpose(2, 3)
    else:
        g = m.reshape(B, K, Y, X)
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    return g


def from_grid(g, lay):
    tr, fy, fx = lay
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    if tr:
        g = g.transpose(2, 3)
    return g.reshape(g.shape[0], -1)


def sym_list(square):
    out = []
    for tr in ([0, 1] if square else [0]):
        for fy in (0, 1):
            for fx in (0, 1):
                if tr or fy or fx:
                    out.append((tr, fy, fx))
    return out


def apply_sym(x, s):
    tr, fy, fx = s
    if tr:
        x = x.transpose(2, 3)
    if fy:
        x = x.flip(2)
    if fx:
        x = x.flip(3)
    return x


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1, padding_mode=pad_mode())
        self.norm = gn(c)
        self.ctx = nn.Linear(2 * c, c) if GLOBAL_CTX else None

    def forward(self, s):
        s = F.relu(s + self.norm(self.conv(s)))
        if self.ctx is not None:
            g = torch.cat([s.mean((2, 3)), s.amax((2, 3))], 1)
            s = s + F.relu(self.ctx(g))[:, :, None, None]
        return s


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.Z = int(spec.Z)
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.n_extra = self.n_actions - self.n_place
        YX = self.Y * self.X
        self.grid = self.n_place > 0 and self.n_place % YX == 0
        self.K = self.n_place // YX
        offs = [0, 1] if DONE_FIRST else [0]
        self.offs = [o for o in offs if o <= self.n_extra]
        self.lays = [tuple(l) for l in LAYOUTS]
        self.syms = sym_list(self.Y == self.X) if OBS_SYM else []
        cin = self.C * (1 + len(self.syms)) + 2
        self.use3d = C3D > 0 and self.C >= 4 * self.Z
        if self.use3d:
            self.c3a = nn.Conv3d(4, C3D, 3, padding=1)
            self.c3b = nn.Conv3d(C3D, C3D // 2, 3, padding=1)
            cin += (C3D // 2) * self.Z
        if self.grid:
            cin += len(self.offs) * len(self.lays) * self.K
        else:
            self.mask_embed = nn.Linear(self.n_actions, EMB * YX)
            cin += EMB
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), gn(STEM), nn.ReLU(),
                                  nn.Conv2d(STEM, WIDTH, 1), gn(WIDTH), nn.ReLU())
        self.blocks = nn.ModuleList([Block(WIDTH), Block(WIDTH)])
        self.hconv = nn.Conv2d(WIDTH, HEAD, HEAD_K, padding=HEAD_K // 2, padding_mode=pad_mode())
        self.hskip = nn.Conv2d(WIDTH, HEAD, 1)
        self.hnorm = gn(HEAD)
        if self.grid:
            self.hout = nn.Conv2d(HEAD, len(self.offs) * len(self.lays) * self.K, 1)
        else:
            self.hred = nn.Conv2d(HEAD, EMB, 1)
            self.hlin = nn.Linear(EMB * YX, self.n_actions)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, max(self.n_extra, 1))
        self.done_first = nn.Linear(64, 1) if (DONE_FIRST and self.n_extra >= 1) else None
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, Z = self.Y, self.X, self.Z
        feats = [obs]
        for s in self.syms:
            feats.append(apply_sym(obs, s))
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        feats.append(yy)
        feats.append(xx)
        if self.use3d:
            v = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
            v = F.relu(self.c3a(v))
            v = F.relu(self.c3b(v))
            feats.append(v.reshape(B, -1, Y, X))
        mf = mask.to(obs.dtype)
        if self.grid:
            for o in self.offs:
                seg = mf[:, o:o + self.n_place]
                for lay in self.lays:
                    feats.append(to_grid(seg, lay, self.K, Y, X))
        else:
            feats.append(self.mask_embed(mf).view(B, -1, Y, X))
        f = self.stem(torch.cat(feats, 1))
        for blk in self.blocks:
            f = blk(f)
        h = F.relu(self.hnorm(self.hconv(f) + self.hskip(f)))
        if self.grid:
            maps = self.hout(h)
            logits = None
            i = 0
            for o in self.offs:
                acc = None
                for lay in self.lays:
                    part = from_grid(maps[:, i * self.K:(i + 1) * self.K], lay)
                    acc = part if acc is None else acc + part
                    i += 1
                padded = F.pad(acc, (o, self.n_extra - o))
                logits = padded if logits is None else logits + padded
        else:
            logits = self.hlin(self.hred(h).flatten(1))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        d = self.done_head(g)[:, :self.n_extra]
        logits = logits + F.pad(d, (self.n_place, 0))
        if self.done_first is not None:
            logits = logits + F.pad(self.done_first(g), (0, self.n_actions - 1))
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
