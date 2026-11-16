import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

STEM = 64
WIDTH = 40
HEAD = 32
EMB = 8
# layout = (transpose, flip_y, flip_x)
LAYOUTS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
DILS = [1, 1]
DENSE = True
NORM = False
GLOBAL_CTX = False
CRITIC = 0
POOLHEAD = False


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


def gn(c):
    return nn.GroupNorm(8, c) if NORM else nn.Identity()


class Block(nn.Module):
    def __init__(self, c, dil):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=dil, dilation=dil)
        self.norm = gn(c)
        self.ctx = nn.Linear(2 * c, c) if GLOBAL_CTX else None

    def forward(self, s):
        s = F.relu(s + self.norm(self.conv(s)))
        if self.ctx is not None:
            g = torch.cat([s.mean((2, 3)), s.amax((2, 3))], 1)
            s = s + F.relu(self.ctx(g))[:, :, None, None]
        return s


class PoolHead(nn.Module):
    def __init__(self, c, fps, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in fps]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Critic(nn.Module):
    def __init__(self, cin, w):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, w, 1), nn.ReLU())
        self.c1 = nn.Conv2d(w, w, 3, padding=1)
        self.c2 = nn.Conv2d(w, w, 3, padding=1)
        self.fc = nn.Sequential(nn.Linear(2 * w, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        s = self.stem(x)
        s = F.relu(s + self.c1(s))
        s = F.relu(s + self.c2(s))
        return self.fc(torch.cat([s.mean((2, 3)), s.amax((2, 3))], 1)).squeeze(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        self.Y = Y
        self.X = X
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.n_extra = self.n_actions - self.n_place
        YX = Y * X
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.crops = [(max(Y - a + 1, 0), max(X - b + 1, 0)) for a, b in fps]
        ncrop = sum(h * w for h, w in self.crops)
        if self.n_place > 0 and self.n_place % YX == 0:
            self.mode = "grid"
            self.K = self.n_place // YX
        elif self.n_place > 0 and self.n_place == ncrop:
            self.mode = "crop"
            self.K = len(fps)
        else:
            self.mode = "dense"
            self.K = 0
        self.lays = [tuple(l) for l in LAYOUTS] if self.mode == "grid" else [(0, 0, 0)]
        use_dense = DENSE or self.mode == "dense"
        cin = self.C + 2
        if self.mode == "grid":
            cin += len(self.lays) * self.K
        elif self.mode == "crop":
            cin += self.K
        if use_dense:
            self.mask_embed = nn.Linear(self.n_actions, EMB * YX)
            cin += EMB
        else:
            self.mask_embed = None
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), gn(STEM), nn.ReLU(),
                                  nn.Conv2d(STEM, WIDTH, 1), gn(WIDTH), nn.ReLU())
        self.blocks = nn.ModuleList([Block(WIDTH, d) for d in DILS])
        pool_ok = self.mode == "grid" and POOLHEAD and len(fps) == self.K
        self.pool = PoolHead(WIDTH, fps, 32, 32) if pool_ok else None
        if self.mode == "grid" and not pool_ok:
            self.hout = nn.Conv2d(HEAD, len(self.lays) * self.K, 1)
        elif self.mode == "crop":
            self.hout = nn.Conv2d(HEAD, self.K, 1)
        else:
            self.hout = None
        if use_dense:
            self.hred = nn.Conv2d(HEAD, EMB, 1)
            self.hlin = nn.Linear(EMB * YX, self.n_place)
        else:
            self.hred = None
            self.hlin = None
        if self.hout is not None or self.hlin is not None:
            self.hconv = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
            self.hskip = nn.Conv2d(WIDTH, HEAD, 1)
            self.hnorm = gn(HEAD)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, max(self.n_extra, 1))
        if CRITIC > 0:
            self.critic = Critic(cin, CRITIC)
            self.value_head = None
        else:
            self.critic = None
            self.value_head = nn.Linear(64, 1)

    def crop_in(self, seg):
        B = seg.shape[0]
        out = []
        off = 0
        for hh, ww in self.crops:
            n = hh * ww
            g = seg[:, off:off + n].reshape(B, 1, hh, ww)
            out.append(F.pad(g, (0, self.X - ww, 0, self.Y - hh)))
            off += n
        return torch.cat(out, 1)

    def crop_out(self, maps):
        out = []
        for t, (hh, ww) in enumerate(self.crops):
            out.append(maps[:, t, :hh, :ww].reshape(maps.shape[0], -1))
        return torch.cat(out, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        mf = mask.to(obs.dtype)
        seg = mf[:, :self.n_place]
        feats = [obs, yy, xx]
        if self.mode == "grid":
            for lay in self.lays:
                feats.append(to_grid(seg, lay, self.K, Y, X))
        elif self.mode == "crop":
            feats.append(self.crop_in(seg))
        if self.mask_embed is not None:
            feats.append(self.mask_embed(mf).view(B, EMB, Y, X))
        x = torch.cat(feats, 1)
        f = self.stem(x)
        for blk in self.blocks:
            f = blk(f)
        place = None
        if self.pool is not None:
            place = self.pool(f)
        if self.hout is not None or self.hlin is not None:
            h = F.relu(self.hnorm(self.hconv(f) + self.hskip(f)))
            if self.hout is not None:
                maps = self.hout(h)
                if self.mode == "grid":
                    for i, lay in enumerate(self.lays):
                        part = from_grid(maps[:, i * self.K:(i + 1) * self.K], lay)
                        place = part if place is None else place + part
                else:
                    part = self.crop_out(maps)
                    place = part if place is None else place + part
            if self.hlin is not None:
                part = self.hlin(self.hred(h).flatten(1))
                place = part if place is None else place + part
        if place is None:
            place = obs.new_zeros(B, self.n_place)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        d = self.done_head(g)[:, :self.n_extra]
        logits = F.pad(place, (0, self.n_extra)) + F.pad(d, (self.n_place, 0))
        if self.critic is not None:
            value = self.critic(x)
        else:
            value = self.value_head(g).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
