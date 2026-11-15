import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 512, "rollout_len": 16, "n_minibatches": 64, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
LC = 32
ZC = 12
PROJ = 16
HID = 16


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
        self.lin = nn.Conv2d(12, LC, 1)
        self.zmix = nn.Conv2d(2 * ZC, LC, 1)
        self.tproj = nn.Conv2d(WIDTH, LC, 1)
        self.ldw = nn.Conv2d(LC, LC, 3, padding=1, groups=LC)
        self.lmix = nn.Conv2d(LC, LC, 1)
        self.hproj = nn.Conv2d(LC, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))

        build = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        b_below = F.pad(build[:, :Z - 1], (0, 0, 0, 0, 1, 0))
        t_below = F.pad(tgt[:, :Z - 1], (0, 0, 0, 0, 1, 0))
        b_above = F.pad(build[:, 1:], (0, 0, 0, 0, 0, 1))
        t_above = F.pad(tgt[:, 1:], (0, 0, 0, 0, 0, 1))
        per = torch.stack([build, tgt, miss, extra, b_below, t_below, b_above, t_above], dim=2)
        gch = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zi = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        li = torch.cat([per, gch, zi], dim=2).reshape(B * Z, 12, Y, X)

        a = self.lin(li)
        c = F.relu(a.view(B, Z, LC, Y, X)[:, :, :ZC])
        up = torch.cummax(c, dim=1).values
        dn = torch.flip(torch.cummax(torch.flip(c, [1]), dim=1).values, [1])
        zc = torch.cat([up, dn], dim=2).reshape(B * Z, 2 * ZC, Y, X)

        t = self.tproj(f).unsqueeze(1).expand(B, Z, LC, Y, X).reshape(B * Z, LC, Y, X)
        h = F.relu(a + self.zmix(zc) + t)
        h = F.relu(h + self.lmix(F.relu(self.ldw(h))))
        p = self.hproj(h)
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
            o = out(hh).view(B, Z, 2, Y, X)
            sc = o[:, :, 0] + F.log_softmax(o[:, :, 1], dim=1)
            res.append(torch.logsumexp(sc, dim=1))
        place = torch.stack(res, dim=1).flatten(1)

        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
