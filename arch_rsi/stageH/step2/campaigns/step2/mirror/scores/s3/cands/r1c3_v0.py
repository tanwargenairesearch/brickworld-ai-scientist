import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 64, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2}

H = 32
P = 24
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        k_in = 7 + 3 + self.Z
        self.enc1 = nn.Conv2d(k_in, H, 1)
        self.enc2 = nn.Conv2d(H, H, 1)
        self.mixz = nn.Conv2d(3 * H, H, 1)
        self.glob_in = nn.Linear(2 * H, H)
        self.ctx = nn.Conv2d(H, H, 3, padding=1)
        self.proj = nn.Conv2d(H, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, P, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(P, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * H, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Z, Y, X = self.Z, self.Y, self.X
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        bb = F.pad(b, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        tb = F.pad(t, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        mb = F.pad(m, (0, 0, 0, 0, 1, 0))[:, :Z]
        loc = torch.stack([b, t, m, e, bb, tb, mb], 2)
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        oh = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        inp = torch.cat([loc, gl, oh], 2).reshape(B * Z, -1, Y, X)
        u = F.relu(self.enc1(inp))
        u = F.relu(self.enc2(u))
        u5 = u.view(B, Z, H, Y, X)
        cs = torch.cumsum(u5, 1)
        below = (cs - u5) / Z
        above = (cs[:, -1:] - cs) / Z
        gu = self.glob_in(torch.cat([u5.mean((1, 3, 4)), u5.amax((1, 3, 4))], 1))
        v = torch.cat([u5, below, above], 2).reshape(B * Z, 3 * H, Y, X)
        v = self.mixz(v).view(B, Z, H, Y, X) + gu.view(B, 1, H, 1, 1)
        v = F.relu(v).reshape(B * Z, H, Y, X)
        v = F.relu(v + self.ctx(v))
        v5 = v.view(B, Z, H, Y, X)
        p = self.proj(v)
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
            s = out(h).view(B, Z, Y * X)
            res.append(torch.logsumexp(s, 1))
        place = torch.cat(res, dim=1)
        g = self.glob(torch.cat([v5.mean((1, 3, 4)), v5.amax((1, 3, 4))], 1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
