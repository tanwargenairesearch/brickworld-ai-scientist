import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 48
W = 32
LW = 24
PROJ = 16
HID = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.layer_in = nn.Conv2d(W + 13, LW, 1)
        self.proj = nn.Conv2d(LW, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W + 2 * LW, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        lay = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        below = F.pad(lay, (0, 0, 0, 0, 1, 0))[:, :, :Z]
        above = F.pad(lay, (0, 0, 0, 0, 0, 1))[:, :, 1:]
        zc = torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1)
        zc = zc.view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        fz = f.unsqueeze(2).expand(B, f.shape[1], Z, Y, X)
        h = torch.cat([fz, lay, below, above, zc], dim=1)
        h = h.permute(0, 2, 1, 3, 4).reshape(B * Z, -1, Y, X)
        h = F.relu(self.layer_in(h))
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
            o = out(F.relu(mix(torch.cat([mx, mn, av], dim=1))))
            o = o.reshape(B, Z, 2, Y, X)
            gate = torch.softmax(o[:, :, 0], dim=1)
            score = (gate * o[:, :, 1]).sum(1)
            res.append(score.flatten(1))
        place = torch.cat(res, dim=1)
        hz = h.reshape(B, Z, LW, Y, X)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)),
                                 hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
