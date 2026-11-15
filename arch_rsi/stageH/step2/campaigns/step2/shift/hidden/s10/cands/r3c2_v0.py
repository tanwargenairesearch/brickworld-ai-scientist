import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.001, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.01}

STEM = 48
WIDTH = 40
NBLK = 3
KL = 8
KS = 8
HID = 24


class LayerHead(nn.Module):
    def __init__(self, c, Z, footprints, k, s, hid):
        super().__init__()
        self.Z = int(Z)
        self.k = k
        self.s = s
        self.hid = hid
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, self.Z * k + s, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * (k + s), hid, 1) for _ in self.fps])
        self.zemb = nn.Parameter(torch.zeros(len(self.fps), self.Z, hid))
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def split(self, t):
        B = t.shape[0]
        Y = t.shape[2]
        X = t.shape[3]
        Z = self.Z
        lay = t[:, :Z * self.k].reshape(B, Z, self.k, Y, X)
        sh = t[:, Z * self.k:].unsqueeze(1).expand(B, Z, self.s, Y, X)
        return torch.cat([lay, sh], dim=2)

    def forward(self, f):
        B = f.shape[0]
        Y = f.shape[2]
        X = f.shape[3]
        Z = self.Z
        p = self.proj(f)
        res = []
        for i, ((dy, dx), mix, out) in enumerate(zip(self.fps, self.mix, self.outs)):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            u = torch.cat([self.split(mx), self.split(mn), self.split(av)], dim=2)
            u = u.reshape(B * Z, 3 * (self.k + self.s), Y, X)
            h = mix(u).reshape(B, Z, self.hid, Y, X) + self.zemb[i][None, :, :, None, None]
            h = F.relu(h).reshape(B * Z, self.hid, Y, X)
            o = out(h).reshape(B, Z, Y, X)
            res.append(torch.logsumexp(o, dim=1))
        return torch.stack(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(NBLK)])
        self.ctx = nn.ModuleList([nn.Linear(2 * WIDTH, WIDTH) for _ in range(NBLK)])
        self.head = LayerHead(WIDTH, spec.Z, spec.footprints, KL, KS, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = self.stem(obs)
        for conv, ctx in zip(self.blocks, self.ctx):
            g = ctx(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
            f = F.relu(f + conv(f) + g[:, :, None, None])
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
