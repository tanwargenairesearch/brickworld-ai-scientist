import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 3, "gae_lambda": 0.95}

W = 32
P = 16
H = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        c_in = 4 * (2 * Z - 1) + 3 + Z
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, W, 1), nn.ReLU(),
            nn.Conv2d(W, W, 1), nn.ReLU())
        self.ctx = nn.Conv2d(W, W, 3, padding=1)
        self.proj = nn.Conv2d(W, P, 1)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * P, H, 1), nn.ReLU(), nn.Conv2d(H, 1, 1))
            for _ in self.footprints])
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        parts = []
        for gi in range(4):
            s = obs[:, gi * Z:(gi + 1) * Z]
            sp = F.pad(s, (0, 0, 0, 0, Z - 1, Z - 1))
            rel = torch.cat([sp[:, z:z + 2 * Z - 1].unsqueeze(1) for z in range(Z)], dim=1)
            parts.append(rel)
        extra = obs[:, 4 * Z:].unsqueeze(1).expand(B, Z, 3, Y, X)
        zp = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        x = torch.cat(parts + [extra, zp], dim=2).reshape(B * Z, -1, Y, X)
        f = self.stem(x)
        f = f + F.relu(self.ctx(f))
        g = F.relu(self.proj(f))
        outs = []
        for fp, head in zip(self.footprints, self.heads):
            dy, dx = fp
            gp = F.pad(g, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(gp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-gp, (dy, dx), stride=1)
            av = F.avg_pool2d(gp, (dy, dx), stride=1)
            o = head(torch.cat([mx, mn, av], dim=1)).reshape(B, Z, Y * X)
            outs.append(torch.logsumexp(o, dim=1))
        place = torch.cat(outs, dim=1)
        fz = f.reshape(B, Z, W, Y, X)
        pooled = torch.cat([fz.mean((1, 3, 4)), fz.amax(4).amax(3).amax(1)], dim=1)
        gl = self.glob(pooled)
        logits = torch.cat([place, self.done_head(gl)], dim=1)
        return logits, self.value_head(gl).squeeze(1)


def build_policy(spec):
    return Net(spec)
