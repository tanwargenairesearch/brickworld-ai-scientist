import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.97, "gae_lambda": 0.92}

WIDTH = 40
STEM = 64
HEADC = 32
HEAD_K = 3
MASK_SKIP = True
OBS_SYM = True
GLOBAL_CTX = True
DILS = [1, 2]
POOL_HEAD = True
FLIPS = [(False, False), (True, False), (False, True), (True, True)]
# layout = (axis order of flat placement index major->minor with 0=type 1=y 2=x, y_flip, x_flip)
LAYOUTS = [
    ((0, 1, 2), False, False), ((0, 1, 2), True, False), ((0, 1, 2), False, True), ((0, 1, 2), True, True),
    ((0, 2, 1), False, False), ((0, 2, 1), True, False), ((0, 2, 1), False, True), ((0, 2, 1), True, True),
    ((1, 2, 0), False, False), ((1, 2, 0), True, False), ((1, 2, 0), False, True), ((1, 2, 0), True, True),
    ((2, 1, 0), False, False), ((2, 1, 0), True, False), ((2, 1, 0), False, True), ((2, 1, 0), True, True),
    ((1, 0, 2), False, False), ((1, 0, 2), True, False), ((1, 0, 2), False, True), ((1, 0, 2), True, True),
    ((2, 0, 1), False, False), ((2, 0, 1), True, False), ((2, 0, 1), False, True), ((2, 0, 1), True, True)]


def to_grid(m, lay, T, Y, X):
    order, fy, fx = lay
    B = m.shape[0]
    sizes = [T, Y, X]
    g = m.reshape(B, sizes[order[0]], sizes[order[1]], sizes[order[2]])
    g = g.permute(0, 1 + order.index(0), 1 + order.index(1), 1 + order.index(2))
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    return g


def from_grid(g, lay):
    order, fy, fx = lay
    B = g.shape[0]
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    g = g.permute(0, 1 + order[0], 1 + order[1], 1 + order[2])
    return g.reshape(B, -1)


def sym_views(obs, sq):
    views = []
    bases = [obs, obs.transpose(2, 3)] if sq else [obs]
    for b in bases:
        for fy, fx in FLIPS:
            v = b
            if fy:
                v = v.flip(2)
            if fx:
                v = v.flip(3)
            views.append(v)
    return views


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
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_actions) - 1
        hw = self.Y * self.X
        self.grid = self.n_place > 0 and self.n_place % hw == 0
        self.T = self.n_place // hw if self.grid else 0
        self.sq = self.Y == self.X
        self.n_sym = (8 if self.sq else 4) if OBS_SYM else 1
        mch = len(LAYOUTS) * self.T if self.grid else 0
        cin = int(spec.in_channels) * self.n_sym + 2 + mch
        if not self.grid:
            self.mask_emb = nn.Linear(self.n_place, 16)
            cin += 16
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(),
                                  nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=d, dilation=d) for d in DILS])
        if GLOBAL_CTX:
            self.gls = nn.ModuleList([nn.Linear(2 * WIDTH, WIDTH) for _ in DILS])
        self.hconv = nn.Conv2d(WIDTH, HEADC, HEAD_K, padding=HEAD_K // 2)
        sk = WIDTH + (mch if MASK_SKIP else 0)
        self.hskip = nn.Conv2d(sk, HEADC, 1)
        if self.grid:
            self.outs = nn.ModuleList([nn.Conv2d(HEADC, self.T, 1) for _ in LAYOUTS])
        else:
            self.fc_red = nn.Conv2d(HEADC, 8, 1)
            self.fc_out = nn.Linear(8 * hw, self.n_place)
        self.use_pool = POOL_HEAD and self.grid and len(spec.footprints) == self.T
        if self.use_pool:
            self.pool = PoolHead(WIDTH, spec.footprints, 32, 32)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        m = mask[:, :self.n_place].to(obs.dtype)
        if self.n_sym > 1:
            parts = sym_views(obs, self.sq)
        else:
            parts = [obs]
        parts = parts + [ys, xs]
        grids = []
        if self.grid:
            grids = [to_grid(m, lay, self.T, Y, X) for lay in LAYOUTS]
            parts = parts + grids
        else:
            e = self.mask_emb(m)
            parts.append(e.view(B, -1, 1, 1).expand(B, e.shape[1], Y, X))
        h = self.stem(torch.cat(parts, 1))
        for i, conv in enumerate(self.blocks):
            z = h + conv(h)
            if GLOBAL_CTX:
                c = self.gls[i](torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))
                z = z + c.view(B, -1, 1, 1)
            h = F.relu(z)
        f = h
        if MASK_SKIP and self.grid:
            sk = torch.cat([f] + grids, 1)
        else:
            sk = f
        hh = F.relu(self.hconv(f) + self.hskip(sk))
        if self.grid:
            place = None
            for lay, out in zip(LAYOUTS, self.outs):
                p = from_grid(out(hh), lay)
                place = p if place is None else place + p
        else:
            place = self.fc_out(F.relu(self.fc_red(hh)).flatten(1))
        if self.use_pool:
            place = place + self.pool(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
