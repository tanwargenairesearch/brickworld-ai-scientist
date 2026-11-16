import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.001, "ent_coef": 0.005}

STEM = 64
COL = 48
WIDTH = 32
LW = 24
PROJ = 16
HID = 24


class LayerPoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
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
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.lin_raw = nn.Conv2d(12 + Z, LW, 1)
        self.lin_col = nn.Conv2d(WIDTH, LW, 1)
        self.l2 = nn.Conv2d(LW, LW, 1)
        self.head = LayerPoolHead(LW, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + LW, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        occ = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 2, 1, 3, 4)
        pad = F.pad(occ, (0, 0, 0, 0, 0, 0, 1, 1))
        below = pad[:, :-2]
        above = pad[:, 2:]
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        inp = torch.cat([below, occ, above, eye], dim=2).reshape(B * Z, 12 + Z, Y, X)
        a = self.lin_raw(inp).view(B, Z, LW, Y, X)
        b = self.lin_col(f).unsqueeze(1)
        h = F.relu(a + b).reshape(B * Z, LW, Y, X)
        h = F.relu(h + self.l2(h))
        sc = self.head(h)
        T = sc.shape[1]
        sc = sc.view(B, Z, T, Y, X)
        place = torch.logsumexp(sc, dim=1).flatten(1)
        hz = h.view(B, Z, LW, Y, X).amax(dim=(1, 3, 4))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), hz], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
