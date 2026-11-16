import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "lr": 0.001, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005}

COLW = 32
LW = 48
FW = 32
PROJ = 16
HID = 24
GH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.col_in = nn.Conv2d(spec.in_channels, COLW, 1)
        self.col_c = nn.Conv2d(COLW, COLW, 3, padding=1)
        self.col_to_layer = nn.Conv2d(COLW, LW, 1)
        self.lay_in = nn.Conv2d(9, LW, 1)
        self.lay2 = nn.Conv2d(LW, FW, 1)
        self.zmix = nn.Conv2d(3 * FW, FW, 1)
        self.proj = nn.Conv2d(FW, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * FW + 2 * COLW, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        bb = F.pad(b, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        tb = F.pad(t, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        ba = F.pad(b, (0, 0, 0, 0, 0, 1))[:, 1:]
        ta = F.pad(t, (0, 0, 0, 0, 0, 1))[:, 1:]
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, Z, 1, 1).expand(B, Z, Y, X)
        lay = torch.stack([b, t, m, e, bb, tb, ba, ta, zc], dim=2).reshape(B * Z, 9, Y, X)

        c = F.relu(self.col_in(obs))
        c = F.relu(c + self.col_c(c))
        cl = self.col_to_layer(c)
        cl = cl.unsqueeze(1).expand(B, Z, LW, Y, X).reshape(B * Z, LW, Y, X)

        h = F.relu(self.lay_in(lay) + cl)
        h = F.relu(self.lay2(h))
        h5 = h.reshape(B, Z, FW, Y, X)
        hb = F.pad(h5, (0, 0, 0, 0, 0, 0, 1, 0))[:, :Z]
        ha = F.pad(h5, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:]
        hz = torch.cat([hb, h5, ha], dim=2).reshape(B * Z, 3 * FW, Y, X)
        f = F.relu(h + self.zmix(hz))

        p = self.proj(f)
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
        T = len(self.fps)
        per = torch.cat(res, dim=1).reshape(B, Z, T, Y, X)
        place = torch.logsumexp(per, dim=1).flatten(1)

        f5 = f.reshape(B, Z, FW, Y, X)
        g = torch.cat([f5.mean((1, 3, 4)), f5.amax((1, 3, 4)), c.mean((2, 3)), c.amax((2, 3))], dim=1)
        g = self.glob(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
