import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

STEM = 64
WIDTH = 40
HP = 32
HK = 3
MEMB = 8
OBS_VIEWS = ["id", "hwc", "xyc"]
OBS_FLAT = 4
OFFSETS = [0, 1]
DENSE = True
GLOBAL_BCAST = True
PADMODE = "zeros"
POOL = True
ANCHORS = ["tl", "br"]
PP = 8
PMIX = 32


def full_layouts(n_place, Y, X):
    P = Y * X
    res = []
    if P <= 0 or n_place <= 0 or n_place % P != 0:
        return res
    T = n_place // P
    for minor in (False, True):
        for tr in (False, True):
            for fy in (False, True):
                for fx in (False, True):
                    cells = []
                    for a in range(n_place):
                        if minor:
                            t = a % T
                            r = a // T
                        else:
                            t = a // P
                            r = a % P
                        if tr:
                            y = r % Y
                            x = r // Y
                        else:
                            y = r // X
                            x = r % X
                        if fy:
                            y = Y - 1 - y
                        if fx:
                            x = X - 1 - x
                        cells.append((t, y, x))
                    res.append((T, cells))
    return res


def compact_layouts(n_place, Y, X, fps):
    res = []
    if len(fps) == 0:
        return res
    for swap in (False, True):
        dims = [(b, a) if swap else (a, b) for a, b in fps]
        sizes = [max(Y - dy + 1, 0) * max(X - dx + 1, 0) for dy, dx in dims]
        if sum(sizes) != n_place:
            continue
        for colmajor in (False, True):
            for br in (False, True):
                cells = []
                for t, (dy, dx) in enumerate(dims):
                    ny = Y - dy + 1
                    nx = X - dx + 1
                    if ny <= 0 or nx <= 0:
                        continue
                    for r in range(ny * nx):
                        if colmajor:
                            u = r % ny
                            v = r // ny
                        else:
                            u = r // nx
                            v = r % nx
                        if br:
                            cells.append((t, u + dy - 1, v + dx - 1))
                        else:
                            cells.append((t, u, v))
                res.append((len(dims), cells))
    return res


def layout_tables(n_place, Y, X, fps):
    P = Y * X
    cand = full_layouts(n_place, Y, X) + compact_layouts(n_place, Y, X, fps)
    seen = set()
    gall = []
    aall = []
    tsum = 0
    for T, cells in cand:
        if len(cells) != n_place:
            continue
        key = tuple(cells)
        if key in seen:
            continue
        g = [n_place] * (T * P)
        ok = True
        aidx = []
        for a, (t, y, x) in enumerate(cells):
            if t < 0 or t >= T or y < 0 or y >= Y or x < 0 or x >= X:
                ok = False
                break
            k = t * P + y * X + x
            if g[k] != n_place:
                ok = False
                break
            g[k] = a
            aidx.append(tsum * P + k)
        if not ok:
            continue
        seen.add(key)
        gall.extend(g)
        aall.append(aidx)
        tsum += T
    return gall, aall, tsum


def coords(obs, Y, X):
    B = obs.shape[0]
    ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
    xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
    return torch.cat([ys, xs], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.P = self.Y * self.X
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        gall, aall, tsum = layout_tables(self.n_place, self.Y, self.X, self.fps)
        self.L = len(aall)
        self.tsum = tsum
        self.use_grid = self.L > 0 and self.n_actions == self.n_place + 1
        self.offsets = list(OFFSETS) if self.use_grid else []
        if self.use_grid:
            self.register_buffer("gidx", torch.tensor(gall, dtype=torch.long))
            self.register_buffer("aidx", torch.tensor(aall, dtype=torch.long).reshape(-1))
        self.dense = DENSE or not self.use_grid
        cin = self.C * len(OBS_VIEWS) + 2 + OBS_FLAT + len(self.offsets) * self.tsum
        if OBS_FLAT > 0:
            self.flat = nn.Linear(self.C * self.P, OBS_FLAT * self.P)
        if self.dense:
            self.dense_in = nn.Linear(self.n_actions, MEMB * self.P)
            cin += MEMB
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1, padding_mode=PADMODE)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1, padding_mode=PADMODE)
        if GLOBAL_BCAST:
            self.gb = nn.Linear(2 * WIDTH, WIDTH)
        self.hconv = nn.Conv2d(WIDTH, HP, HK, padding=HK // 2, padding_mode=PADMODE)
        self.hskip = nn.Conv2d(WIDTH, HP, 1)
        hc = HP
        if POOL:
            self.pproj = nn.Conv2d(WIDTH, PP, 1)
            self.pmix = nn.Conv2d(len(self.fps) * len(ANCHORS) * 2 * PP, PMIX, 1)
            hc += PMIX
        if self.use_grid:
            self.outc = nn.Conv2d(hc, len(self.offsets) * self.tsum, 1)
        if self.dense:
            self.dense_proj = nn.Conv2d(hc, 4, 1)
            self.dense_out = nn.Linear(4 * self.P, self.n_actions)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(max(len(self.offsets), 1))])
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, C = self.Y, self.X, self.C
        m = mask.float()
        parts = []
        for v in OBS_VIEWS:
            if v == "hwc":
                parts.append(obs.reshape(B, Y, X, C).permute(0, 3, 1, 2))
            elif v == "xyc":
                parts.append(obs.reshape(B, X, Y, C).permute(0, 3, 2, 1))
            else:
                parts.append(obs)
        parts.append(coords(obs, Y, X))
        if OBS_FLAT > 0:
            parts.append(self.flat(obs.reshape(B, -1)).view(B, OBS_FLAT, Y, X))
        for off in self.offsets:
            mp = m[:, off:off + self.n_place]
            mz = torch.cat([mp, m.new_zeros(B, 1)], dim=1)
            parts.append(mz[:, self.gidx].view(B, self.tsum, Y, X))
        if self.dense:
            parts.append(self.dense_in(m).view(B, MEMB, Y, X))
        x = self.stem(torch.cat(parts, dim=1))
        x = F.relu(x + self.r1(x))
        if GLOBAL_BCAST:
            x = x + self.gb(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))[:, :, None, None]
        f = F.relu(x + self.r2(x))
        h = F.relu(self.hconv(f) + self.hskip(f))
        if POOL:
            q = self.pproj(f)
            feats = []
            for dy, dx in self.fps:
                for an in ANCHORS:
                    if dy * dx == 1:
                        mx = q
                        av = q
                    else:
                        if an == "br":
                            pad = (dx - 1, 0, dy - 1, 0)
                        else:
                            pad = (0, dx - 1, 0, dy - 1)
                        qq = F.pad(q, pad)
                        mx = F.max_pool2d(qq, (dy, dx), stride=1)
                        av = F.avg_pool2d(qq, (dy, dx), stride=1)
                    feats.append(mx)
                    feats.append(av)
            h = torch.cat([h, F.relu(self.pmix(torch.cat(feats, dim=1)))], dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = h.new_zeros(B, self.n_actions)
        if self.use_grid:
            outs = self.outc(h).flatten(1)
            blk = self.tsum * self.P
            for i, off in enumerate(self.offsets):
                o = outs[:, i * blk:(i + 1) * blk]
                place = o[:, self.aidx].view(B, self.L, self.n_place).sum(1)
                d = self.done_heads[i](g)
                if off == 0:
                    part = torch.cat([place, d], dim=1)
                else:
                    part = torch.cat([d, place], dim=1)
                logits = logits + part
        else:
            logits = logits + F.pad(self.done_heads[0](g), (self.n_actions - 1, 0))
        if self.dense:
            logits = logits + self.dense_out(self.dense_proj(h).flatten(1))
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
