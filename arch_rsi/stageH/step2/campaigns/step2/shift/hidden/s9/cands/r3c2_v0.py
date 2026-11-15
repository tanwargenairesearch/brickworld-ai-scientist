import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
D = 24
PROJ = 32
HID = 32

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "gamma": 0.98, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.001}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        Z = self.Z
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.layer_in = nn.Linear(4, D)
        self.zemb = nn.Embedding(Z, D)
        self.ctx = nn.Conv2d(WIDTH, D, 1)
        self.layer_mid = nn.Linear(D, D)
        self.hgt = nn.Conv2d(WIDTH, Z + 1, 1)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        nin = 6 * D + 3 * PROJ + 1
        self.mix = nn.ModuleList([nn.Conv2d(nin, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))

        per = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 2, 3, 4, 1)
        e = self.layer_in(per)
        ctx = self.ctx(f).permute(0, 2, 3, 1).unsqueeze(1)
        zid = torch.arange(Z, device=obs.device)
        ze = self.zemb(zid).view(1, Z, 1, 1, D)
        g = F.relu(e + ctx + ze)
        g = F.relu(self.layer_mid(g))
        g = g.permute(0, 1, 4, 2, 3).reshape(B, Z * D, Y, X)

        hp = torch.softmax(self.hgt(f), dim=1)
        logc = torch.log(torch.cumsum(hp, dim=1).clamp(1e-6, 1.0))
        p = self.proj(f)

        res = []
        for k, (dy, dx) in enumerate(self.fps):
            if dy * dx == 1:
                lc = logc
                gmx, gmn, gav = g, g, g
                pmx, pmn, pav = p, p, p
            else:
                pd = (0, dx - 1, 0, dy - 1)
                lc = F.avg_pool2d(F.pad(logc, pd), (dy, dx), stride=1) * float(dy * dx)
                gq = F.pad(g, pd)
                gmx = F.max_pool2d(gq, (dy, dx), stride=1)
                gmn = -F.max_pool2d(-gq, (dy, dx), stride=1)
                gav = F.avg_pool2d(gq, (dy, dx), stride=1)
                pq = F.pad(p, pd)
                pmx = F.max_pool2d(pq, (dy, dx), stride=1)
                pmn = -F.max_pool2d(-pq, (dy, dx), stride=1)
                pav = F.avg_pool2d(pq, (dy, dx), stride=1)
            cm = torch.exp(lc)
            pl = cm - F.pad(cm[:, :-1], (0, 0, 0, 0, 1, 0))
            wl = pl[:, :Z].unsqueeze(2)
            wb = (1.0 - cm[:, :Z]).clamp_min(0.0).unsqueeze(2) / float(Z)
            gp = torch.cat([gmx.view(B, Z, D, Y, X), gmn.view(B, Z, D, Y, X),
                            gav.view(B, Z, D, Y, X)], dim=2)
            land = (wl * gp).sum(1)
            below = (wb * gp).sum(1)
            feat = torch.cat([land, below, pl[:, Z:Z + 1], pmx, pmn, pav], dim=1)
            h = F.relu(self.mix[k](feat))
            res.append(self.outs[k](h))
        place = torch.cat(res, dim=1).flatten(1)

        gl = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(gl)], dim=1)
        return logits, self.value_head(gl).squeeze(1)


def build_policy(spec):
    return Net(spec)
