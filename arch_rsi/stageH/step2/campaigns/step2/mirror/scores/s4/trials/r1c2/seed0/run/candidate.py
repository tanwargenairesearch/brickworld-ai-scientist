import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 256, "rollout_len": 8, "n_minibatches": 16, "ent_coef": 0.005}

D = 24
P = 12
H = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.inp = nn.Conv2d(16, D, 1)
        self.c1 = nn.Conv2d(D, D, 3, padding=1)
        self.zmix = nn.Conv2d(3 * D, D, 1)
        self.proj = nn.Conv2d(D, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        v = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        vp = F.pad(v, (0, 0, 0, 0, 1, 1))
        nb = torch.cat([vp[:, :, 0:Z], vp[:, :, 1:Z + 1], vp[:, :, 2:Z + 2]], dim=1)
        g = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        x = torch.cat([nb, g, zc], dim=1)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * Z, 16, Y, X)
        h = F.relu(self.inp(x))
        h = F.relu(h + self.c1(h))
        hz = h.reshape(B, Z, D, Y, X)
        zero = hz.new_zeros(B, 1, D, Y, X)
        hp = torch.cat([zero, hz, zero], dim=1)
        n = torch.cat([hp[:, 0:Z], hp[:, 1:Z + 1], hp[:, 2:Z + 2]], dim=2).reshape(B * Z, 3 * D, Y, X)
        h = F.relu(h + self.zmix(n))
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
            w = torch.softmax(o[:, :, 0], dim=1)
            res.append((w * o[:, :, 1]).sum(1).flatten(1))
        place = torch.cat(res, dim=1)
        hf = h.reshape(B, Z, D, Y, X)
        gf = torch.cat([hf.mean((1, 3, 4)), hf.amax((1, 3, 4))], dim=1)
        gg = self.glob(gf)
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
