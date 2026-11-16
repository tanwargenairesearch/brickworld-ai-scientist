import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 5e-4, "ent_coef": 0.03, "ppo_epochs": 1, "n_envs": 64, "rollout_len": 8, "n_minibatches": 8}

H = 24
P = 12
HID = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        nf = 9 + 3 + self.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.inp = nn.Conv2d(nf, H, 1)
        self.c1 = nn.Conv2d(H, H, 3, padding=1)
        self.cross = nn.Conv2d(3 * H, H, 1)
        self.proj = nn.Conv2d(H, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * H, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        b_lo = F.pad(b, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        b_hi = F.pad(b, (0, 0, 0, 0, 0, 1))[:, 1:]
        t_lo = F.pad(t, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        t_hi = F.pad(t, (0, 0, 0, 0, 0, 1))[:, 1:]
        m_lo = F.pad(m, (0, 0, 0, 0, 1, 0))[:, :Z]
        per = torch.stack([b, b_lo, b_hi, t, t_lo, t_hi, m, m_lo, e], dim=2)
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype).reshape(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        x = torch.cat([per, gl, eye], dim=2).reshape(B * Z, -1, Y, X)
        h = F.relu(self.inp(x))
        h = F.relu(h + self.c1(h))
        hv = h.reshape(B, Z, H, Y, X)
        mean = hv.mean(1, keepdim=True).expand(B, Z, H, Y, X)
        mxl = hv.amax(1, keepdim=True).expand(B, Z, H, Y, X)
        cin = torch.cat([hv, mean, mxl], dim=2).reshape(B * Z, 3 * H, Y, X)
        h = F.relu(h + self.cross(cin))
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
            score = o[:, :, 0]
            w = torch.softmax(o[:, :, 1], dim=1)
            res.append((w * score).sum(1))
        place = torch.stack(res, dim=1).flatten(1)
        hv = h.reshape(B, Z, H, Y, X)
        g = self.glob(torch.cat([hv.mean((1, 3, 4)), hv.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
