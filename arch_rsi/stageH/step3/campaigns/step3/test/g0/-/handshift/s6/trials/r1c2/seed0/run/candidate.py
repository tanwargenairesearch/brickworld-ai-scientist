import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32}

TW = 32
LH = 32
LP = 16
MH = 16
NIN = 11


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        c = spec.in_channels
        self.stem = nn.Sequential(nn.Conv2d(c, 48, 1), nn.ReLU(), nn.Conv2d(48, TW, 1), nn.ReLU())
        self.c1 = nn.Conv2d(TW, TW, 3, padding=1)
        self.c2 = nn.Conv2d(TW, TW, 3, padding=1)
        self.layer_in = nn.Conv2d(NIN, LH, 1)
        self.col_in = nn.Conv2d(TW, LH, 1)
        self.layer_proj = nn.Conv2d(LH, LP, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * LP, MH, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(MH, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * TW, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        build = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        bb = F.pad(build, (0, 0, 0, 0, 1, 0))[:, :Z]
        tb = F.pad(tgt, (0, 0, 0, 0, 1, 0))[:, :Z]
        per = torch.stack([build, tgt, miss, extra, bb, tb], dim=2)
        hts = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zr = torch.arange(Z, device=obs.device, dtype=obs.dtype)
        zc = (zr + 0.5) / Z
        g0 = (zr == 0).to(obs.dtype)
        pos = torch.stack([zc, g0], dim=1).view(1, Z, 2, 1, 1).expand(B, Z, 2, Y, X)
        inp = torch.cat([per, hts, pos], dim=2).reshape(B * Z, NIN, Y, X)

        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))

        col = self.col_in(f).unsqueeze(1).expand(B, Z, LH, Y, X).reshape(B * Z, LH, Y, X)
        h = F.relu(self.layer_in(inp) + col)
        p = self.layer_proj(h)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            hh = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(hh))
        sc = torch.cat(res, dim=1).reshape(B, Z, -1)
        place = torch.logsumexp(sc, dim=1)

        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
