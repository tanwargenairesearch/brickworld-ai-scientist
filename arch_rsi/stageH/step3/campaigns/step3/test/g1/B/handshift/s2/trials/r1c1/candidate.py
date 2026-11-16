import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

STEM = 64
WIDTH = 40
HEAD = 32
HEAD_K = 3
EMB = 8
# layout = (transpose, flip_y, flip_x, type_minor)
LAYOUTS = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)]
DONE_FIRST = False
OBS_VIEWS = False


def to_grid(m, lay, K, Y, X):
    tr, fy, fx, minor = lay
    B = m.shape[0]
    if tr:
        if minor:
            g = m.reshape(B, X, Y, K).permute(0, 3, 2, 1)
        else:
            g = m.reshape(B, K, X, Y).transpose(2, 3)
    else:
        if minor:
            g = m.reshape(B, Y, X, K).permute(0, 3, 1, 2)
        else:
            g = m.reshape(B, K, Y, X)
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    return g


def from_grid(g, lay):
    tr, fy, fx, minor = lay
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    if tr:
        g = g.transpose(2, 3)
    if minor:
        g = g.permute(0, 2, 3, 1)
    return g.reshape(g.shape[0], -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
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
        self.n_views = 3 if OBS_VIEWS else 1
        cin = self.C * self.n_views + 2
        if self.grid:
            cin += len(self.offs) * len(self.lays) * self.K
        else:
            self.mask_embed = nn.Linear(self.n_actions, EMB * YX)
            cin += EMB
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(),
                                  nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hconv = nn.Conv2d(WIDTH, HEAD, HEAD_K, padding=HEAD_K // 2)
        self.hskip = nn.Conv2d(WIDTH, HEAD, 1)
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
        Y, X = self.Y, self.X
        feats = [obs]
        if self.n_views > 1:
            feats.append(obs.reshape(B, Y, X, self.C).permute(0, 3, 1, 2))
            feats.append(obs.reshape(B, self.C, X, Y).transpose(2, 3))
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        feats.append(yy)
        feats.append(xx)
        mf = mask.to(obs.dtype)
        if self.grid:
            for o in self.offs:
                seg = mf[:, o:o + self.n_place]
                for lay in self.lays:
                    feats.append(to_grid(seg, lay, self.K, Y, X))
        else:
            feats.append(self.mask_embed(mf).view(B, -1, Y, X))
        s = self.stem(torch.cat(feats, 1))
        s = F.relu(s + self.c1(s))
        f = F.relu(s + self.c2(s))
        h = F.relu(self.hconv(f) + self.hskip(f))
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
