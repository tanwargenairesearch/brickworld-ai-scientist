import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 3}

H = 40
PROJ = 16
HID = 16
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = spec.in_channels + 12 + 1 + Z
        self.l1 = nn.Conv2d(cin, H, 1)
        self.l2 = nn.Conv2d(H, H, 1)
        self.proj = nn.Conv2d(H, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * H, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        per = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], dim=2)
        zero = torch.zeros_like(per[:, :1])
        below = torch.cat([zero, per[:, :-1]], dim=1)
        above = torch.cat([per[:, 1:], zero], dim=1)
        o = obs.unsqueeze(1).expand(B, Z, C, Y, X)
        zpos = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        x = torch.cat([o, below, per, above, zpos, eye], dim=2).reshape(B * Z, -1, Y, X)
        h = F.relu(self.l1(x))
        h = F.relu(h + self.l2(h))
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
            r = out(F.relu(mix(torch.cat([mx, mn, av], dim=1))))
            r = r.view(B, Z, 2, Y, X)
            s = r[:, :, 0]
            w = torch.softmax(r[:, :, 1], dim=1)
            res.append((w * s).sum(1))
        place = torch.stack(res, dim=1).flatten(1)
        hv = h.view(B, Z, H, Y, X)
        gf = torch.cat([hv.mean((1, 3, 4)), hv.amax((1, 3, 4))], dim=1)
        g = self.glob(gf)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
