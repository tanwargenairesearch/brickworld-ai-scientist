import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16}

TR = 40
LH = 48
LF = 32
PROJ = 16
HID = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = spec.in_channels
        self.stem = nn.Sequential(
            nn.Conv2d(C, 64, 1), nn.ReLU(),
            nn.Conv2d(64, TR, 1), nn.ReLU())
        self.c1 = nn.Conv2d(TR, TR, 3, padding=1)
        self.c2 = nn.Conv2d(TR, TR, 3, padding=1)
        self.local = nn.Conv2d(12, LH, 1)
        self.colp = nn.Conv2d(C, LH, 1)
        self.ctxp = nn.Conv2d(TR, LH, 1)
        self.emb = nn.Embedding(self.Z, LH)
        self.l2 = nn.Conv2d(LH, LF, 1)
        self.proj = nn.Conv2d(LF, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * TR + 2 * LF, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        lay = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        lp = F.pad(lay, (0, 0, 0, 0, 1, 1))
        own = lp[:, :, 1:Z + 1]
        below = lp[:, :, 0:Z]
        above = lp[:, :, 2:Z + 2]
        loc = torch.cat([own, below, above], dim=1)
        loc = loc.permute(0, 2, 1, 3, 4).reshape(B * Z, 12, Y, X)
        h = self.local(loc).reshape(B, Z, LH, Y, X)
        cell = self.colp(obs) + self.ctxp(f)
        e = self.emb(torch.arange(Z, device=obs.device))
        h = h + cell.unsqueeze(1) + e.view(1, Z, LH, 1, 1)
        h = F.relu(h).reshape(B * Z, LH, Y, X)
        g = F.relu(self.l2(h))
        p = self.proj(g)
        scores = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            hh = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            sc = out(hh).reshape(B, Z, Y, X)
            scores.append(torch.logsumexp(sc, dim=1))
        place = torch.stack(scores, dim=1).flatten(1)
        gz = g.reshape(B, Z, LF, Y, X)
        gfeat = torch.cat([f.mean((2, 3)), f.amax((2, 3)),
                           gz.mean((1, 3, 4)), gz.amax((1, 3, 4))], dim=1)
        gg = self.glob(gfeat)
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
