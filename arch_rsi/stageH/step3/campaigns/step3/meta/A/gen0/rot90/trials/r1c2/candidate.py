import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.003}

STEM = 64
COL = 48
WIDTH = 32
D = 16
P = 12
HID = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.enc = nn.Conv2d(7 + self.Z, D, 1)
        self.ctx = nn.Conv2d(WIDTH, D, 1)
        self.proj = nn.Conv2d(2 * D, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(6 * P, HID, 1) for _ in self.fps])
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

        per = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 2, 1, 3, 4)
        shared = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        pos = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        inp = torch.cat([per, shared, pos], dim=2).reshape(B * Z, 7 + Z, Y, X)
        e = self.enc(inp).view(B, Z, D, Y, X)
        e = F.relu(e + self.ctx(f).unsqueeze(1))
        cm = e.flip(1).cummax(1).values.flip(1)
        h = torch.cat([e, cm], dim=2).reshape(B * Z, 2 * D, Y, X)
        p = self.proj(h)

        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            pooled = torch.cat([mx, mn, av], dim=1).reshape(B, Z, 3 * P, Y, X)
            below = torch.cat([torch.zeros_like(pooled[:, :1]), pooled[:, :-1]], dim=1)
            u = torch.cat([pooled, below], dim=2).reshape(B * Z, 6 * P, Y, X)
            sc = out(F.relu(mix(u))).reshape(B, Z, Y, X)
            res.append(torch.logsumexp(sc, dim=1))
        place = torch.stack(res, dim=1).flatten(1)

        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
