import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 3, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

OBS_SYM = True
HEAD_K = 3
HEAD_W = 32
STEM = 64
WIDTH = 40
NRES = 2


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.N = int(spec.n_actions)
        self.P = self.N - 1
        YX = self.Y * self.X
        self.grid = (self.P % YX == 0)
        self.K = self.P // YX if self.grid else 0
        syms = []
        if OBS_SYM:
            for tr in ((0, 1) if self.Y == self.X else (0,)):
                for fy in (0, 1):
                    for fx in (0, 1):
                        syms.append((tr, fy, fx))
        else:
            syms.append((0, 0, 0))
        self.osyms = syms
        cin = int(spec.in_channels) * len(syms) + 2
        if self.grid:
            cin += self.K
        else:
            self.mask_embed = nn.Linear(self.N, 8 * YX)
            cin += 8
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(NRES)])
        self.hconv = nn.Conv2d(WIDTH, HEAD_W, HEAD_K, padding=HEAD_K // 2)
        self.hskip = nn.Conv2d(WIDTH, HEAD_W, 1)
        if self.grid:
            self.hout = nn.Conv2d(HEAD_W, self.K, 1)
        else:
            self.hred = nn.Conv2d(HEAD_W, 8, 1)
            self.hlin = nn.Linear(8 * YX, self.P)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        feats = []
        for tr, fy, fx in self.osyms:
            o = obs
            if tr:
                o = o.transpose(2, 3)
            if fy:
                o = o.flip(2)
            if fx:
                o = o.flip(3)
            feats.append(o)
        m = mask.float()
        if self.grid:
            feats.append(m[:, :self.P].reshape(B, self.K, Y, X))
        else:
            feats.append(self.mask_embed(m).view(B, 8, Y, X))
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device).view(1, 1, 1, X).expand(B, 1, Y, X)
        feats.append(yy)
        feats.append(xx)
        f = self.stem(torch.cat(feats, dim=1))
        for conv in self.res:
            f = F.relu(f + conv(f))
        h = F.relu(self.hconv(f) + self.hskip(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        d = self.done_head(g)
        if self.grid:
            place = self.hout(h).reshape(B, -1)
        else:
            place = self.hlin(F.relu(self.hred(h)).flatten(1))
        logits = torch.cat([place, d], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
