import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 256, "rollout_len": 16, "n_minibatches": 64, "lr": 0.0007, "ent_coef": 0.005, "gamma": 0.97, "gae_lambda": 0.9}

TW = 32
H1 = 32
P = 16
M = 16
GH = 64


def foot_pool(q, dy, dx):
    if dy * dx == 1:
        return torch.cat([q, q, q], dim=1)
    r = F.pad(q, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(r, (dy, dx), stride=1)
    mn = -F.max_pool2d(-r, (dy, dx), stride=1)
    av = F.avg_pool2d(r, (dy, dx), stride=1)
    return torch.cat([mx, mn, av], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = (int(spec.in_channels) - 3) // 4
        self.Z = Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.t0 = nn.Conv2d(spec.in_channels, TW, 1)
        self.t1 = nn.Conv2d(TW, TW, 3, padding=1)
        self.t2 = nn.Conv2d(TW, TW, 3, padding=1)
        self.lin = nn.Linear(12 + 3 + 1 + Z, H1)
        self.ctx = nn.Linear(TW, H1)
        self.lout = nn.Linear(H1, P)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, M, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(M, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * TW + 2 * H1, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        f = F.relu(self.t0(obs))
        f = F.relu(self.t1(f))
        f = F.relu(f + self.t2(f))
        parts = []
        for k in range(4):
            a = obs[:, k * Z:(k + 1) * Z]
            p = F.pad(a, (0, 0, 0, 0, 1, 1))
            parts += [p[:, 0:Z], p[:, 1:Z + 1], p[:, 2:Z + 2]]
        lay = torch.stack(parts, dim=2)
        hts = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = ((torch.arange(Z, device=obs.device, dtype=obs.dtype) + 0.5) / Z).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        u = torch.cat([lay, hts, zc, eye], dim=2).permute(0, 1, 3, 4, 2)
        c = self.ctx(f.permute(0, 2, 3, 1)).unsqueeze(1)
        h = F.relu(self.lin(u) + c)
        q = self.lout(h).permute(0, 1, 4, 2, 3).reshape(B * Z, P, Y, X)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            m = F.relu(mix(foot_pool(q, dy, dx)))
            o = out(m).reshape(B, Z, 2, Y, X)
            w = torch.softmax(o[:, :, 1], dim=1)
            res.append((w * o[:, :, 0]).sum(1))
        place = torch.stack(res, dim=1).flatten(1)
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3)), h.mean((1, 2, 3)), h.amax((1, 2, 3))], dim=1)
        g = self.glob(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
