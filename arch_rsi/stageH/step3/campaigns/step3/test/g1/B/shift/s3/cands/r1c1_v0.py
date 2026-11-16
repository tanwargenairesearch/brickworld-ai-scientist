import torch
import torch.nn as nn
import torch.nn.functional as F

LAYOUTS = [(0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
HEAD_K = 3
DONE_BOTH = False
OBS_SYM = False
STEM = 64
WIDTH = 40
HEADC = 32
RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}


def decode(v, lay, T, Y, X):
    minor, tr, fy, fx = lay
    B = v.shape[0]
    if minor:
        g = v.reshape(B, X, Y, T) if tr else v.reshape(B, Y, X, T)
        g = g.permute(0, 3, 1, 2)
    else:
        g = v.reshape(B, T, X, Y) if tr else v.reshape(B, T, Y, X)
    if tr:
        g = g.transpose(2, 3)
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    return g


def encode(m, lay):
    minor, tr, fy, fx = lay
    if fy:
        m = m.flip(2)
    if fx:
        m = m.flip(3)
    if tr:
        m = m.transpose(2, 3)
    if minor:
        m = m.permute(0, 2, 3, 1)
    return m.reshape(m.shape[0], -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y, self.X, self.T = int(spec.Y), int(spec.X), int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.ok = (self.n_place == self.T * self.Y * self.X) and (self.n_actions == self.n_place + 1)
        self.sq = self.Y == self.X
        self.shifts = [0, 1] if (DONE_BOTH and self.ok) else [0]
        copies = 1
        if OBS_SYM:
            copies = 4 if self.sq else 3
        cin = spec.in_channels * copies + 2
        if self.ok:
            cin += len(self.shifts) * len(LAYOUTS) * self.T
        else:
            self.mask_embed = nn.Linear(self.n_actions, 4 * self.Y * self.X)
            cin += 4
        ys = torch.linspace(-1, 1, self.Y).view(1, 1, self.Y, 1).expand(1, 1, self.Y, self.X)
        xs = torch.linspace(-1, 1, self.X).view(1, 1, 1, self.X).expand(1, 1, self.Y, self.X)
        self.register_buffer("coords", torch.cat([ys, xs], 1).clone())
        self.stem1 = nn.Conv2d(cin, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hconv = nn.Conv2d(WIDTH, HEADC, HEAD_K, padding=HEAD_K // 2)
        self.hskip = nn.Conv2d(WIDTH, HEADC, 1)
        if self.ok:
            self.n_maps = len(self.shifts) * len(LAYOUTS)
            self.outs = nn.Conv2d(HEADC, self.n_maps * self.T, 1)
        else:
            self.fb_red = nn.Conv2d(HEADC, 8, 1)
            self.fb_out = nn.Linear(8 * self.Y * self.X, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, len(self.shifts))
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask.float()
        feats = [obs]
        if OBS_SYM:
            feats.append(obs.flip(2))
            feats.append(obs.flip(3))
            if self.sq:
                feats.append(obs.transpose(2, 3))
        if self.ok:
            for s in self.shifts:
                v = m[:, s:s + self.n_place]
                for lay in LAYOUTS:
                    feats.append(decode(v, lay, self.T, self.Y, self.X))
        else:
            feats.append(self.mask_embed(m).reshape(B, 4, self.Y, self.X))
        feats.append(self.coords.expand(B, -1, -1, -1))
        x = torch.cat(feats, 1)
        x = F.relu(self.stem1(x))
        x = F.relu(self.stem2(x))
        f = F.relu(x + self.r1(x))
        f = F.relu(f + self.r2(f))
        h = F.relu(self.hconv(f) + self.hskip(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        d = self.done_head(g)
        if self.ok:
            maps = self.outs(h).split(self.T, dim=1)
            logits = None
            k = 0
            for i, s in enumerate(self.shifts):
                p = None
                for lay in LAYOUTS:
                    e = encode(maps[k], lay)
                    k += 1
                    p = e if p is None else p + e
                di = d[:, i:i + 1]
                part = torch.cat([p, di], 1) if s == 0 else torch.cat([di, p], 1)
                logits = part if logits is None else logits + part
        else:
            p = self.fb_out(F.relu(self.fb_red(h)).flatten(1))
            logits = torch.cat([p, d[:, :1]], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
