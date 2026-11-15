import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 64, "rollout_len": 8, "n_minibatches": 8, "ppo_epochs": 2, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
LP = 24
LH = 16


class LayerHead(nn.Module):
    def __init__(self, spec, c):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        in_l = c + 6 + 3 + self.Z
        self.p1 = nn.Conv2d(in_l, LP, 1)
        self.p2 = nn.Conv2d(2 * LP, LP, 1)
        self.convs = nn.ModuleList([nn.Conv2d(LP, LH, (dy, dx)) for dy, dx in self.fps])
        self.mix = nn.ModuleList([nn.Conv2d(3 * LP, LH, 1, bias=False) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(LH, 1, 1) for _ in self.fps])

    def forward(self, f, obs):
        B, _, Y, X = obs.shape
        Z = self.Z
        build = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        rest = obs[:, 4 * Z:4 * Z + 3]
        bb = F.pad(build, (0, 0, 0, 0, 1, 0))[:, :Z]
        tb = F.pad(tgt, (0, 0, 0, 0, 1, 0))[:, :Z]
        per = torch.stack([build, tgt, miss, extra, bb, tb], dim=2)
        onehot = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        restx = rest.unsqueeze(1).expand(B, Z, 3, Y, X)
        fx = f.unsqueeze(1).expand(B, Z, f.shape[1], Y, X)
        inp = torch.cat([per, restx, onehot, fx], dim=2).reshape(B * Z, -1, Y, X)
        h = F.relu(self.p1(inp))
        hz = h.reshape(B, Z, LP, Y, X)
        cm = torch.cummax(hz, dim=1)[0]
        below = torch.cat([torch.zeros_like(cm[:, :1]), cm[:, :-1]], dim=1)
        p = self.p2(torch.cat([hz, below], dim=2).reshape(B * Z, 2 * LP, Y, X))
        res = []
        for (dy, dx), conv, mix, out in zip(self.fps, self.convs, self.mix, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            hh = F.relu(conv(q) + mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(hh))
        s = torch.cat(res, dim=1).reshape(B, Z, len(self.fps), Y, X)
        s = torch.logsumexp(s, dim=1)
        return s.flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = LayerHead(spec, WIDTH)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f, obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
