import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 8, "n_minibatches": 8}

H = 20
P = 16
M = 12


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.inp1 = nn.Conv2d(8, H, 1)
        self.inp2 = nn.Conv2d(H, H, 1)
        self.zmix1 = nn.Conv2d(2 * H, H, (3, 1), padding=(1, 0))
        self.sp1 = nn.Conv2d(H, H, 3, padding=1)
        self.zmix2 = nn.Conv2d(2 * H, H, (3, 1), padding=(1, 0))
        self.proj = nn.Conv2d(H, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, M, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(M, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * H, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def above(self, h):
        r = torch.flip(h, [2])
        r = torch.cummax(r, dim=2).values
        return torch.flip(r, [2])

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        N = Y * X
        per = obs[:, :4 * Z].reshape(B, 4, Z, N)
        gl = obs[:, 4 * Z:4 * Z + 3].reshape(B, 3, 1, N).expand(B, 3, Z, N)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) + 0.5) / Z
        zc = zc.view(1, 1, Z, 1).expand(B, 1, Z, N)
        u = torch.cat([per, gl, zc], dim=1)
        h = F.relu(self.inp1(u))
        h = F.relu(self.inp2(h))
        h = F.relu(h + self.zmix1(torch.cat([h, self.above(h)], dim=1)))
        t = h.permute(0, 2, 1, 3).reshape(B * Z, H, Y, X)
        t = self.sp1(t).reshape(B, Z, H, N).permute(0, 2, 1, 3)
        h = F.relu(h + t)
        h = F.relu(h + self.zmix2(torch.cat([h, self.above(h)], dim=1)))
        p = self.proj(h)
        p = p.permute(0, 2, 1, 3).reshape(B * Z, P, Y, X)
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
            o = o.reshape(B, Z, 2, N)
            w = torch.softmax(o[:, :, 0], dim=1)
            res.append((w * o[:, :, 1]).sum(1))
        place = torch.cat(res, dim=1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
