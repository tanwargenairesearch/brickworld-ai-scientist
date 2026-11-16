import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

FACT = True
TREL = True
CIRC = True
STEM = 64
WIDTH = 40
HEAD = 32
GH = 64
EMB = 8


def make_layouts(fact):
    poss = [0, 1, 2] if fact else [0, 1]
    out = []
    for p in poss:
        for tr in (0, 1):
            for fy in (0, 1):
                for fx in (0, 1):
                    out.append((p, tr, fy, fx))
    return out


def to_grid(v, lay, K, Y, X):
    p, tr, fy, fx = lay
    B = v.shape[0]
    if p == 0:
        g = v.reshape(B, K, X, Y).transpose(2, 3) if tr else v.reshape(B, K, Y, X)
    elif p == 1:
        g = v.reshape(B, X, Y, K).permute(0, 3, 2, 1) if tr else v.reshape(B, Y, X, K).permute(0, 3, 1, 2)
    else:
        g = v.reshape(B, X, K, Y).permute(0, 2, 3, 1) if tr else v.reshape(B, Y, K, X).permute(0, 2, 1, 3)
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    return g


def from_grid(g, lay):
    p, tr, fy, fx = lay
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    if p == 0:
        if tr:
            g = g.transpose(2, 3)
    elif p == 1:
        g = g.permute(0, 3, 2, 1) if tr else g.permute(0, 2, 3, 1)
    else:
        g = g.permute(0, 3, 1, 2) if tr else g.permute(0, 2, 1, 3)
    return g.reshape(g.shape[0], -1)


def compact_in(v, fps, Y, X, tr):
    B = v.shape[0]
    chans = []
    off = 0
    for dy, dx in fps:
        h, w = Y - dy + 1, X - dx + 1
        seg = v[:, off:off + h * w]
        off += h * w
        if tr:
            g = seg.reshape(B, w, h).transpose(1, 2)
        else:
            g = seg.reshape(B, h, w)
        chans.append(F.pad(g, (0, X - w, 0, Y - h)).unsqueeze(1))
    return torch.cat(chans, 1)


def compact_out(o, fps, Y, X, tr):
    res = []
    for t, (dy, dx) in enumerate(fps):
        h, w = Y - dy + 1, X - dx + 1
        g = o[:, t, :h, :w]
        if tr:
            g = g.transpose(1, 2)
        res.append(g.reshape(o.shape[0], -1))
    return torch.cat(res, 1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y, self.X, self.Z = int(spec.Y), int(spec.X), int(spec.Z)
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        Y, X, Z = self.Y, self.X, self.Z
        YX = Y * X
        self.mode = 'dense'
        self.K = 0
        if self.n_place > 0 and self.n_place % YX == 0 and (FACT or self.n_place // YX == int(spec.n_types)):
            self.mode = 'grid'
            self.K = self.n_place // YX
        elif FACT:
            okf = all(Y - dy + 1 >= 1 and X - dx + 1 >= 1 for dy, dx in self.fps)
            cnt = sum((Y - dy + 1) * (X - dx + 1) for dy, dx in self.fps)
            if okf and cnt == self.n_place:
                self.mode = 'compact'
                self.K = len(self.fps)
        self.lays = make_layouts(FACT) if self.mode == 'grid' else []
        self.tsyms = []
        self.tsplit = self.C == 4 * Z + 3
        tch = (Z + 1) if self.tsplit else self.C
        if TREL:
            for tr in (0, 1):
                for fy in (0, 1):
                    for fx in (0, 1):
                        if (tr, fy, fx) == (0, 0, 0):
                            continue
                        if tr and Y != X:
                            continue
                        self.tsyms.append((tr, fy, fx))
        cin = self.C + 2 + tch * len(self.tsyms)
        if self.mode == 'grid':
            cin += self.K * len(self.lays)
        elif self.mode == 'compact':
            cin += 2 * self.K
        else:
            self.memb = nn.Linear(self.n_place, EMB * YX)
            cin += EMB
        pm = 'circular' if CIRC else 'zeros'
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1, padding_mode=pm)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1, padding_mode=pm)
        self.hk = nn.Conv2d(WIDTH, HEAD, 3, padding=1, padding_mode=pm)
        self.hs = nn.Conv2d(WIDTH, HEAD, 1)
        if self.mode == 'grid':
            self.out = nn.Conv2d(HEAD, self.K * len(self.lays), 1)
        elif self.mode == 'compact':
            self.out = nn.Conv2d(HEAD, 2 * self.K, 1)
        else:
            self.fo = nn.Conv2d(HEAD, 4, 1)
            self.fl = nn.Linear(4 * YX, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done = nn.Linear(GH, 1)
        self.value = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, Z, K = self.Y, self.X, self.Z, self.K
        mp = mask.float()[:, :self.n_place]
        feats = [obs]
        if self.tsyms:
            if self.tsplit:
                tg = torch.cat([obs[:, Z:2 * Z], obs[:, 4 * Z + 1:4 * Z + 2]], 1)
            else:
                tg = obs
            for tr, fy, fx in self.tsyms:
                g = tg
                if tr:
                    g = g.transpose(2, 3)
                if fy:
                    g = g.flip(2)
                if fx:
                    g = g.flip(3)
                feats.append(g)
        if self.mode == 'grid':
            for lay in self.lays:
                feats.append(to_grid(mp, lay, K, Y, X))
        elif self.mode == 'compact':
            feats.append(compact_in(mp, self.fps, Y, X, False))
            feats.append(compact_in(mp, self.fps, Y, X, True))
        else:
            feats.append(self.memb(mp).view(B, EMB, Y, X))
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        feats.append(yy)
        feats.append(xx)
        x = self.stem(torch.cat(feats, 1))
        x = F.relu(x + self.r1(x))
        x = F.relu(x + self.r2(x))
        h = F.relu(self.hk(x) + self.hs(x))
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))
        if self.mode == 'grid':
            o = self.out(h)
            place = 0
            for i, lay in enumerate(self.lays):
                place = place + from_grid(o[:, i * K:(i + 1) * K], lay)
        elif self.mode == 'compact':
            o = self.out(h)
            place = compact_out(o[:, :K], self.fps, Y, X, False) + compact_out(o[:, K:], self.fps, Y, X, True)
        else:
            place = self.fl(F.relu(self.fo(h)).flatten(1))
        logits = torch.cat([place, self.done(g)], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
