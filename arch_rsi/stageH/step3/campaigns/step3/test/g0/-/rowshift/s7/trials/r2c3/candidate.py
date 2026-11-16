import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 64, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "lr": 0.001, "ent_coef": 0.005, "gae_lambda": 0.95}

TR = 32
LH = 24
HID = 24
G = 64
NPER = 9


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        Z = self.Z
        self.trunk = nn.Sequential(
            nn.Conv2d(spec.in_channels, 64, 1), nn.ReLU(),
            nn.Conv2d(64, TR, 1), nn.ReLU())
        self.t_c = nn.Conv2d(TR, TR, 3, padding=1)
        self.l_in = nn.Conv2d(NPER + 3 + Z, LH, 1)
        self.l_tr = nn.Conv2d(TR, LH, 1)
        self.l_c = nn.Conv2d(LH, LH, 3, padding=1)
        self.heads = nn.ModuleList([nn.Conv2d(LH, HID, (dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * LH + 2 * TR, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        t = self.trunk(obs)
        t = F.relu(t + self.t_c(t))
        build = obs[:, 0:Z]
        target = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        bdown = F.pad(build, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        tdown = F.pad(target, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        bup = F.pad(build, (0, 0, 0, 0, 0, 1))[:, 1:]
        mdown = F.pad(miss, (0, 0, 0, 0, 1, 0))[:, :Z]
        mup = F.pad(miss, (0, 0, 0, 0, 0, 1))[:, 1:]
        per = torch.stack([build, target, miss, extra, bdown, bup, mdown, tdown, mup], dim=2)
        g3 = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        lin = torch.cat([per, g3, eye], dim=2).reshape(B * Z, NPER + 3 + Z, Y, X)
        tr = self.l_tr(t).unsqueeze(1).expand(B, Z, LH, Y, X).reshape(B * Z, LH, Y, X)
        h = F.relu(self.l_in(lin) + tr)
        h = F.relu(h + self.l_c(h))
        res = []
        for (dy, dx), hd, out in zip(self.fps, self.heads, self.outs):
            if dy * dx > 1:
                q = F.pad(h, (0, dx - 1, 0, dy - 1))
            else:
                q = h
            o = out(F.relu(hd(q))).reshape(B, Z, Y, X)
            res.append(torch.logsumexp(o, dim=1))
        place = torch.stack(res, dim=1).flatten(1)
        hz = h.reshape(B, Z, LH, Y, X)
        gf = torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4)), t.mean((2, 3)), t.amax((2, 3))], dim=1)
        g = self.glob(gf)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
