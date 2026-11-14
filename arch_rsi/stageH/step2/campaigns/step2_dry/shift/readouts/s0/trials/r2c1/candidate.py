import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "ent_coef": 0.02, "n_minibatches": 16}

STEM = 48
COL = 32
WIDTH = 32
D = 24
P = 12
HD = 24


def roll(t):
    s = t.shape[-1] // 2
    return torch.cat([t[..., s:], t[..., :s]], dim=-1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(2 * spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        lin = 24 + 6 + Z
        self.l1 = nn.Conv2d(lin, D, 1)
        self.l2 = nn.Conv2d(D, D, 1)
        self.tproj = nn.Conv2d(WIDTH, D, 1)
        self.cross = nn.Conv2d(3 * D, D, 1)
        self.proj = nn.Conv2d(D, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, HD, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HD, 2, 1) for _ in self.fps])

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        ro = roll(obs)
        s = self.stem(torch.cat([obs, ro], dim=1))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))

        L = torch.cat([obs[:, :4 * Z], ro[:, :4 * Z]], dim=1).reshape(B, 8, Z, Y, X)
        Lp = F.pad(L, (0, 0, 0, 0, 1, 1))
        below = Lp[:, :, :Z]
        above = Lp[:, :, 2:]
        gch = torch.cat([obs[:, 4 * Z:], ro[:, 4 * Z:]], dim=1).unsqueeze(2).expand(B, 6, Z, Y, X)
        pos = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        inp = torch.cat([L, below, above, gch, pos], dim=1)
        inp = inp.permute(0, 2, 1, 3, 4).reshape(B * Z, -1, Y, X)

        h = F.relu(self.l1(inp))
        t = self.tproj(f).unsqueeze(1).expand(B, Z, D, Y, X).reshape(B * Z, D, Y, X)
        h = F.relu(self.l2(h) + t)
        h5 = h.reshape(B, Z, D, Y, X)
        z0 = h5.new_zeros(B, 1, D, Y, X)
        dn = torch.cat([z0, h5[:, :-1]], dim=1)
        up = torch.cat([h5[:, 1:], z0], dim=1)
        hx = torch.cat([h5, dn, up], dim=2).reshape(B * Z, 3 * D, Y, X)
        h = F.relu(h + self.cross(hx))

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
            o = out(F.relu(mix(torch.cat([mx, mn, av], dim=1)))).reshape(B, Z, 2, Y, X)
            w = torch.softmax(o[:, :, 1], dim=1)
            res.append((w * o[:, :, 0]).sum(1).flatten(1))
        place = torch.cat(res, dim=1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
