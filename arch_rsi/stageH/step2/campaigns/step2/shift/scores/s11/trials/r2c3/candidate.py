import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 64, "rollout_len": 16, "n_minibatches": 8, "ent_coef": 0.005}

CW = 32
W = 32
HID = 24
NLOC = 9


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.col = nn.Sequential(
            nn.Conv2d(spec.in_channels, 64, 1), nn.ReLU(),
            nn.Conv2d(64, CW, 1), nn.ReLU())
        self.ctx = nn.Conv2d(CW, CW, 3, padding=1)
        self.hgt = nn.Conv2d(CW, Z + 1, 1)
        self.g2l = nn.Conv2d(CW, W, 1)
        self.lin = nn.Conv2d(NLOC + Z, W, 1)
        self.win = nn.ModuleList([nn.Conv2d(W, HID, (dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * CW, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        e = self.col(obs)
        f = F.relu(e + self.ctx(e))
        ph = F.softmax(self.hgt(f), dim=1)
        lcdf = torch.log(torch.cumsum(ph, dim=1).clamp(min=1e-6))[:, :Z]
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        ex = obs[:, 3 * Z:4 * Z]
        bb = F.pad(b, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        tb = F.pad(t, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        mb = F.pad(m, (0, 0, 0, 0, 1, 0), value=0.0)[:, :Z]
        ba = F.pad(b, (0, 0, 0, 0, 0, 1), value=0.0)[:, 1:]
        ta = F.pad(t, (0, 0, 0, 0, 0, 1), value=0.0)[:, 1:]
        loc = torch.stack([b, t, m, ex, bb, tb, mb, ba, ta], dim=2)
        oh = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        loc = torch.cat([loc, oh], dim=2).reshape(B * Z, NLOC + Z, Y, X)
        g = self.g2l(f).unsqueeze(1).expand(B, Z, W, Y, X).reshape(B * Z, W, Y, X)
        h = F.relu(self.lin(loc) + g)
        res = []
        for (dy, dx), win, out in zip(self.fps, self.win, self.outs):
            q = F.pad(h, (0, dx - 1, 0, dy - 1))
            s = out(F.relu(win(q))).view(B, Z, Y, X)
            if dy * dx == 1:
                pl = lcdf
            else:
                lq = F.pad(lcdf, (0, dx - 1, 0, dy - 1))
                pl = F.avg_pool2d(lq, (dy, dx), stride=1) * float(dy * dx)
            cdf_l = torch.exp(pl)
            prev = F.pad(cdf_l, (0, 0, 0, 0, 1, 0), value=0.0)[:, :Z]
            p_l = (cdf_l - prev).clamp(min=1e-6)
            res.append(torch.logsumexp(s + torch.log(p_l), dim=1))
        place = torch.stack(res, dim=1).flatten(1)
        gg = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
