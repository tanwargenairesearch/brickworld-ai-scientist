import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "ent_coef": 0.003, "gamma": 0.9, "gae_lambda": 0.85}

STEM = 64
WIDTH = 40
HEADC = 32
HEAD_K = 3
NRES = 2
MAX_CH = 192
OBS_REMAP = False
GLOBAL_BCAST = False
DONE_FIRST = False


def fit(v, H, W):
    v = v[:, :, :H, :W]
    ph = H - v.shape[2]
    pw = W - v.shape[3]
    if ph > 0 or pw > 0:
        v = F.pad(v, (0, pw, 0, ph))
    return v


def grid_variants(K, H, W, exact):
    vs = [("major", 0, 0, 0), ("major", 1, 0, 0), ("minor", 0, 0, 0), ("minor", 1, 0, 0)]
    if exact:
        vs = vs + [("major", 0, 1, 0), ("major", 0, 0, 1)]
    return [{"kind": "grid", "k": K, "H": H, "W": W, "order": o, "tr": t, "fy": a, "fx": b} for o, t, a, b in vs]


def find_layouts(Y, X, P, fps):
    exact, near, valid = [], [], []
    cands = []
    for H in range(max(1, Y - 2), Y + 3):
        for W in range(max(1, X - 2), X + 3):
            if P % (H * W) == 0:
                cands.append((abs(H - Y) + abs(W - X), H, W, P // (H * W)))
    cands.sort()
    for d, H, W, K in cands:
        if d == 0:
            exact += grid_variants(K, H, W, True)
        else:
            near += grid_variants(K, H, W, False)
    seen = []
    for swap in (0, 1):
        dims = []
        for dy, dx in fps:
            a, b = (dx, dy) if swap else (dy, dx)
            dims.append((max(Y - a + 1, 1), max(X - b + 1, 1)))
        if len(dims) > 0 and sum(h * w for h, w in dims) == P and dims not in seen:
            seen.append(dims)
            valid.append({"kind": "valid", "k": len(dims), "dims": dims})
    lays, tot = [], 0
    for lay in exact + valid + near:
        if lays and tot + lay["k"] > MAX_CH:
            continue
        lays.append(lay)
        tot += lay["k"]
    return lays


def to_canon(m, lay, Y, X):
    B = m.shape[0]
    if lay["kind"] == "valid":
        parts, off = [], 0
        for h, w in lay["dims"]:
            seg = m[:, off:off + h * w].reshape(B, 1, h, w)
            parts.append(fit(seg, Y, X))
            off += h * w
        return torch.cat(parts, dim=1)
    K, H, W = lay["k"], lay["H"], lay["W"]
    if lay["order"] == "major":
        if lay["tr"]:
            v = m.reshape(B, K, W, H).transpose(2, 3)
        else:
            v = m.reshape(B, K, H, W)
    else:
        if lay["tr"]:
            v = m.reshape(B, W, H, K).permute(0, 3, 2, 1)
        else:
            v = m.reshape(B, H, W, K).permute(0, 3, 1, 2)
    if lay["fy"]:
        v = v.flip(2)
    if lay["fx"]:
        v = v.flip(3)
    return fit(v, Y, X)


def from_canon(o, lay):
    B = o.shape[0]
    if lay["kind"] == "valid":
        outs = []
        for t, hw in enumerate(lay["dims"]):
            outs.append(fit(o[:, t:t + 1], hw[0], hw[1]).reshape(B, -1))
        return torch.cat(outs, dim=1)
    o = fit(o, lay["H"], lay["W"])
    if lay["fy"]:
        o = o.flip(2)
    if lay["fx"]:
        o = o.flip(3)
    if lay["order"] == "major":
        if lay["tr"]:
            o = o.transpose(2, 3)
    else:
        if lay["tr"]:
            o = o.permute(0, 3, 2, 1)
        else:
            o = o.permute(0, 2, 3, 1)
    return o.reshape(B, -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y, self.X = int(spec.Y), int(spec.X)
        self.C = int(spec.in_channels)
        self.P = int(spec.n_place)
        self.NA = int(spec.n_actions)
        self.n_extra = self.NA - self.P
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.lays = find_layouts(self.Y, self.X, self.P, fps)
        self.shifts = [0, 1] if (DONE_FIRST and self.n_extra == 1) else [0]
        nk = sum(lay["k"] for lay in self.lays)
        if self.lays:
            nm = nk * len(self.shifts)
        else:
            self.mask_emb = nn.Linear(self.NA, 4 * self.Y * self.X)
            nm = 4
        cin = self.C + (self.C if OBS_REMAP else 0) + nm + 2
        if OBS_REMAP:
            self.remap = nn.Linear(self.Y * self.X, self.Y * self.X)
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(NRES)])
        self.hc = nn.Conv2d(WIDTH, HEADC, HEAD_K, padding=HEAD_K // 2)
        self.hs = nn.Conv2d(WIDTH, HEADC, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        if GLOBAL_BCAST:
            self.gb = nn.Linear(64, HEADC)
        if self.lays:
            self.out = nn.Conv2d(HEADC, nm, 1)
        else:
            self.out = nn.Conv2d(HEADC, 4, 1)
            self.fb = nn.Linear(4 * self.Y * self.X, self.P)
        self.done = nn.Linear(64, max(1, self.n_extra))
        self.done2 = nn.Linear(64, 1)
        self.value = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        m = mask.float()
        parts = [obs]
        if OBS_REMAP:
            parts.append(self.remap(obs.reshape(B, self.C, Y * X)).reshape(B, self.C, Y, X))
        if self.lays:
            for s in self.shifts:
                mp = m[:, s:s + self.P]
                for lay in self.lays:
                    parts.append(to_canon(mp, lay, Y, X))
        else:
            parts.append(self.mask_emb(m).reshape(B, 4, Y, X))
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        parts.append(yy)
        parts.append(xx)
        x = self.stem(torch.cat(parts, dim=1))
        for conv in self.res:
            x = F.relu(x + conv(x))
        f = x
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = self.hc(f) + self.hs(f)
        if GLOBAL_BCAST:
            h = h + self.gb(g)[:, :, None, None]
        h = F.relu(h)
        o = self.out(h)
        if self.lays:
            logits = 0
            idx = 0
            for s in self.shifts:
                place = 0
                for lay in self.lays:
                    k = lay["k"]
                    place = place + from_canon(o[:, idx:idx + k], lay)
                    idx += k
                if s == 0:
                    if self.n_extra > 0:
                        full = torch.cat([place, self.done(g)], dim=1)
                    else:
                        full = place
                else:
                    full = torch.cat([self.done2(g), place], dim=1)
                logits = logits + full
        else:
            place = self.fb(o.flatten(1))
            if self.n_extra > 0:
                logits = torch.cat([place, self.done(g)], dim=1)
            else:
                logits = place
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
