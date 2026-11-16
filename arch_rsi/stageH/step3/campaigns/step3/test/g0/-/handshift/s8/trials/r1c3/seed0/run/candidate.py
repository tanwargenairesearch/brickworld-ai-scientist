import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.0015}

H1 = 48
H2 = 32
PROJ = 24
HID = 16


class LayerNet(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = spec.in_channels + 4 + self.Z
        self.l1 = nn.Conv2d(cin, H1, 1)
        self.l2 = nn.Conv2d(H1, H2, 1)
        self.proj = nn.Conv2d(H2, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * H2, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        per = torch.stack([obs[:, k * Z:(k + 1) * Z] for k in range(4)], dim=2)
        col = obs.unsqueeze(1).expand(B, Z, C, Y, X)
        oh = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        inp = torch.cat([col, per, oh], dim=2).reshape(B * Z, C + 4 + Z, Y, X)
        h = F.relu(self.l1(inp))
        h = F.relu(self.l2(h))
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
            s = s.view(B, Z, Y, X)
            res.append(torch.logsumexp(s, dim=1).flatten(1))
        place = torch.cat(res, dim=1)
        hz = h.view(B, Z, H2, Y, X)
        g = self.glob(torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return LayerNet(spec)
