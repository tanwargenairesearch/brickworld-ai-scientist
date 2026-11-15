import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 3, "gamma": 0.97}

TR = 48
TW = 32
LW = 24
PROJ = 16
HID = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = int(spec.in_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(C, TR, 1), nn.ReLU(),
            nn.Conv2d(TR, TW, 1), nn.ReLU())
        self.c1 = nn.Conv2d(TW, TW, 3, padding=1)
        self.c2 = nn.Conv2d(TW, TW, 3, padding=1)
        self.ctx = nn.Conv2d(TW, LW, 1)
        self.lin = nn.Conv2d(8 + C + self.Z, LW, 1)
        self.l2 = nn.Conv2d(LW, LW, 1)
        self.proj = nn.Conv2d(LW, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * TW + LW, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        lay = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        bt = lay[:, :2]
        below = F.pad(bt, (0, 0, 0, 0, 1, 0), value=1.0)[:, :, :Z]
        above = F.pad(bt, (0, 0, 0, 0, 0, 1), value=0.0)[:, :, 1:]
        per = torch.cat([lay, below, above], 1).permute(0, 2, 1, 3, 4)
        full = obs.unsqueeze(1).expand(B, Z, C, Y, X)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype).reshape(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        inp = torch.cat([per, full, eye], 2).reshape(B * Z, 8 + C + Z, Y, X)
        c = self.ctx(f).unsqueeze(1).expand(B, Z, LW, Y, X).reshape(B * Z, LW, Y, X)
        h = F.relu(self.lin(inp) + c)
        h = h + F.relu(self.l2(h))
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
            hh = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(hh))
        sc = torch.cat(res, dim=1)
        T = sc.shape[1]
        sc = sc.reshape(B, Z, T, Y, X)
        place = torch.logsumexp(sc, dim=1).flatten(1)
        hl = h.reshape(B, Z, LW, Y, X).mean((1, 3, 4))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), hl], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
