import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

FEAT = 32
PROJ = 16
HID = 32
GH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        Z = self.Z
        self.zconv = nn.Conv2d(5, FEAT, (2 * Z - 1, 1), padding=(Z - 1, 0))
        self.gconv = nn.Conv2d(3, FEAT, 1)
        self.zmix = nn.Conv2d(FEAT, FEAT, 1)
        self.proj = nn.Conv2d(FEAT, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * FEAT, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Z, Y, X = self.Z, self.Y, self.X
        per = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], dim=1)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) + 1.0) / Z
        zc = zc.view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        u = torch.cat([per, zc], dim=1).reshape(B, 5, Z, Y * X)
        gl = self.gconv(obs[:, 4 * Z:4 * Z + 3]).reshape(B, FEAT, 1, Y * X)
        h = F.relu(self.zconv(u) + gl)
        h = F.relu(h + self.zmix(h))
        p = self.proj(h)
        p = p.permute(0, 2, 1, 3).reshape(B * Z, PROJ, Y, X)
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
        s = torch.cat(res, dim=1).reshape(B, Z, len(self.fps), Y * X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        g = self.glob(pooled)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
