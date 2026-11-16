import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 4, "lr": 0.001, "gamma": 0.98, "gae_lambda": 0.95}

H = 40
PROJ = 16
HID = 24
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = spec.in_channels + 8 + self.Z
        self.col = nn.Sequential(
            nn.Conv2d(cin, H, 1), nn.ReLU(),
            nn.Conv2d(H, H, 1), nn.ReLU())
        self.proj = nn.Conv2d(H, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * H, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        build = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        bb = F.pad(build, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        tb = F.pad(tgt, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        ba = F.pad(build, (0, 0, 0, 0, 0, 1))[:, 1:]
        ta = F.pad(tgt, (0, 0, 0, 0, 0, 1))[:, 1:]
        per = torch.stack([build, tgt, miss, extra, bb, tb, ba, ta], dim=2)
        allc = obs.unsqueeze(1).expand(B, Z, C, Y, X)
        zp = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        x = torch.cat([per, allc, zp], dim=2).reshape(B * Z, -1, Y, X)
        h = self.col(x)
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
            s = out(F.relu(mix(torch.cat([mx, mn, av], dim=1))))
            s = s.reshape(B, Z, Y * X)
            res.append(torch.logsumexp(s, dim=1))
        place = torch.cat(res, dim=1)
        hz = h.reshape(B, Z, -1, Y, X)
        g = self.glob(torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
