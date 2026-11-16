import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
D = 16
H = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.X = int(spec.X)
        self.Y = int(spec.Y)
        self.Z = int(spec.Z)
        Z = self.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        nt = len(self.fps)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.register_buffer("zpos", torch.eye(Z).reshape(1, Z, Z, 1, 1))
        self.tok = nn.Linear(4 + 2 + Z, D)
        self.ctx = nn.Linear(WIDTH, D)
        self.tctx = nn.Conv2d(WIDTH, H * nt, 1)
        self.mix = nn.ModuleList([nn.Linear(3 * D, H) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Linear(H, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Z, Y, X = self.Z, self.Y, self.X
        f = F.relu(self.c1(self.stem(obs)))
        layers = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        hts = obs[:, 4 * Z:4 * Z + 2].reshape(B, 2, 1, Y, X).expand(B, 2, Z, Y, X)
        zp = self.zpos.expand(B, Z, Z, Y, X)
        tok = torch.cat([layers, hts, zp], dim=1).permute(0, 2, 3, 4, 1)
        fc = f.permute(0, 2, 3, 1)
        e = F.relu(self.tok(tok) + self.ctx(fc).unsqueeze(1))
        cum = torch.cummax(e, dim=1).values
        below = torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-1]], dim=1)
        e2 = e.permute(0, 1, 4, 2, 3).reshape(B * Z, D, Y, X)
        b2 = below.permute(0, 1, 4, 2, 3).reshape(B * Z, D, Y, X)
        tc = self.tctx(f)
        res = []
        for i, ((dy, dx), mix, out) in enumerate(zip(self.fps, self.mix, self.outs)):
            if dy * dx == 1:
                mx, mn, bm = e2, e2, b2
            else:
                pad = (0, dx - 1, 0, dy - 1)
                q = F.pad(e2, pad)
                r = F.pad(b2, pad)
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                bm = F.max_pool2d(r, (dy, dx), stride=1)
            p = torch.cat([mx, mn, bm], dim=1).reshape(B, Z, 3 * D, Y, X).permute(0, 3, 4, 1, 2)
            c = tc[:, i * H:(i + 1) * H].permute(0, 2, 3, 1).unsqueeze(3)
            h = F.relu(mix(p) + c)
            s = out(h).squeeze(-1)
            res.append(torch.logsumexp(s, dim=-1).reshape(B, Y * X))
        place = torch.cat(res, dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
