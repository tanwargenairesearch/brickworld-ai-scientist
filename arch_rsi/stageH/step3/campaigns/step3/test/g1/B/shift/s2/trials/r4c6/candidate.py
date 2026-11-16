import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0015, "n_envs": 64, "rollout_len": 16, "n_minibatches": 4,
          "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.95, "gae_lambda": 0.9}

LAYOUTS = (("TYX", 0, 0), ("TYX", 1, 0), ("TYX", 0, 1), ("TYX", 1, 1),
           ("TXY", 0, 0), ("TXY", 1, 0), ("TXY", 0, 1), ("TXY", 1, 1),
           ("YXT", 0, 0), ("YXT", 1, 0), ("YXT", 0, 1), ("YXT", 1, 1),
           ("XYT", 0, 0), ("XYT", 1, 0), ("XYT", 0, 1), ("XYT", 1, 1),
           ("YTX", 0, 0), ("YTX", 1, 0), ("YTX", 0, 1), ("YTX", 1, 1),
           ("XTY", 0, 0), ("XTY", 1, 0), ("XTY", 0, 1), ("XTY", 1, 1))
MASK_IN = True
SPLIT = False
SMALL_INIT = False
AXIAL = False
POOLHEAD = False
DONE_SEP = False
DILS = (1, 2, 4)
STEM = 64
WIDTH = 40
HPROJ = 32
AXW = 16
AXES = {"T": 1, "Y": 2, "X": 3}


def lay_in(m, lay, T, Y, X):
    order, fy, fx = lay
    size = {"T": T, "Y": Y, "X": X}
    B = m.shape[0]
    m = m.reshape(B, size[order[0]], size[order[1]], size[order[2]])
    m = m.permute(0, 1 + order.index("T"), 1 + order.index("Y"), 1 + order.index("X"))
    if fy:
        m = m.flip(2)
    if fx:
        m = m.flip(3)
    return m


def lay_out(L, lay):
    order, fy, fx = lay
    if fy:
        L = L.flip(2)
    if fx:
        L = L.flip(3)
    L = L.permute(0, AXES[order[0]], AXES[order[1]], AXES[order[2]])
    return L.reshape(L.shape[0], -1)


def coords(B, Y, X, ref):
    yy = torch.linspace(-1.0, 1.0, Y, device=ref.device, dtype=ref.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
    xx = torch.linspace(-1.0, 1.0, X, device=ref.device, dtype=ref.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
    return [yy, xx]


class Res(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=d, dilation=d)

    def forward(self, x):
        return F.relu(x + self.conv(x))


class Axial(nn.Module):
    def __init__(self, c, Y, X):
        super().__init__()
        self.row = nn.Conv2d(c, AXW, (1, 2 * X - 1), padding=(0, X - 1))
        self.col = nn.Conv2d(c, AXW, (2 * Y - 1, 1), padding=(Y - 1, 0))
        self.mix = nn.Conv2d(2 * AXW, c, 1)

    def forward(self, x):
        a = F.relu(torch.cat([self.row(x), self.col(x)], 1))
        return F.relu(x + self.mix(a))


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
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


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        T, Y, X = spec.n_types, spec.Y, spec.X
        C = spec.in_channels
        self.T, self.Y, self.X = T, Y, X
        self.n_place = spec.n_place
        self.n_actions = spec.n_actions
        self.ok = spec.n_place == T * Y * X
        self.layouts = tuple(LAYOUTS) if self.ok else ()
        self.nmaps = len(self.layouts) * T
        self.mask_in = MASK_IN and self.ok
        n_m = self.nmaps if self.mask_in else 0
        self.dense = not self.ok
        if self.dense:
            self.mask_remap = nn.Linear(spec.n_actions, 4 * Y * X)
            n_m += 4
        self.n_m = n_m
        if SPLIT:
            self.ostem = nn.Sequential(nn.Conv2d(C + 2, STEM, 1), nn.GroupNorm(8, STEM))
            if n_m > 0:
                self.mstem = nn.Sequential(nn.Conv2d(n_m, STEM, 1), nn.GroupNorm(8, STEM))
            self.stem2 = nn.Sequential(nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        else:
            self.stem = nn.Sequential(nn.Conv2d(C + 2 + n_m, STEM, 1), nn.ReLU(),
                                      nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        blocks = [Res(WIDTH, d) for d in DILS]
        if AXIAL:
            blocks.append(Axial(WIDTH, Y, X))
        self.blocks = nn.Sequential(*blocks)
        self.hconv = nn.Conv2d(WIDTH, HPROJ, 3, padding=1)
        self.hskip = nn.Conv2d(WIDTH, HPROJ, 1)
        if self.nmaps > 0:
            self.out = nn.Conv2d(HPROJ, self.nmaps, 1)
        self.use_ph = POOLHEAD and self.ok and len(spec.footprints) == T
        if self.use_ph:
            self.phead = PoolHead(WIDTH, spec.footprints, 32, 32)
        if self.dense:
            self.oproj = nn.Conv2d(HPROJ, 4, 1)
            self.out_remap = nn.Linear(4 * Y * X, spec.n_actions)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        if DONE_SEP:
            self.dnet = nn.Sequential(nn.Conv2d(C + 2, 32, 1), nn.ReLU(),
                                      nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
            self.dlin = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        if SMALL_INIT:
            if self.nmaps > 0:
                nn.init.orthogonal_(self.out.weight, 0.01)
                nn.init.zeros_(self.out.bias)
            nn.init.orthogonal_(self.done_head.weight, 0.01)
            nn.init.zeros_(self.done_head.bias)
            if self.use_ph:
                for o in self.phead.outs:
                    nn.init.orthogonal_(o.weight, 0.01)
                    nn.init.zeros_(o.bias)
            if DONE_SEP:
                nn.init.orthogonal_(self.dlin[2].weight, 0.01)
                nn.init.zeros_(self.dlin[2].bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        T, Y, X = self.T, self.Y, self.X
        m = mask.to(obs.dtype)
        oin = torch.cat([obs] + coords(B, Y, X, obs), 1)
        mfeats = []
        if self.mask_in:
            for lay in self.layouts:
                mfeats.append(lay_in(m[:, :self.n_place], lay, T, Y, X))
        if self.dense:
            mfeats.append(self.mask_remap(m).reshape(B, 4, Y, X))
        if SPLIT:
            s = self.ostem(oin)
            if self.n_m > 0:
                s = s + self.mstem(torch.cat(mfeats, 1))
            s = self.stem2(s)
        else:
            s = self.stem(torch.cat([oin] + mfeats, 1))
        f = self.blocks(s)
        h = F.relu(self.hconv(f) + self.hskip(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        done = self.done_head(g)
        if DONE_SEP:
            d = self.dnet(oin)
            done = done + self.dlin(torch.cat([d.mean((2, 3)), d.amax((2, 3))], 1))
        if self.nmaps > 0:
            o = self.out(h)
            place = 0
            for k, lay in enumerate(self.layouts):
                place = place + lay_out(o[:, k * T:(k + 1) * T], lay)
            if self.use_ph:
                place = place + self.phead(f)
            logits = torch.cat([place, done], 1)
        else:
            logits = torch.zeros(B, self.n_actions, device=obs.device, dtype=obs.dtype)
        if self.dense:
            logits = logits + self.out_remap(self.oproj(h).flatten(1))
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
