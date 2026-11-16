import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "lr": 0.001, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "n_minibatches": 32}

E = 32
P = 16
H = 32
GH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = 12 + int(spec.in_channels) + self.Z
        self.register_buffer("zid", torch.eye(self.Z))
        self.cell = nn.Sequential(
            nn.Conv2d(cin, 48, 1), nn.ReLU(),
            nn.Conv2d(48, E, 1), nn.ReLU())
        self.proj = nn.Conv2d(E, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(P * dy * dx, H, 1) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * E, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        C = obs.shape[1]
        Y = obs.shape[2]
        X = obs.shape[3]
        Z = self.Z
        L = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 2, 1, 3, 4)
        zero = obs.new_zeros(B, 1, 4, Y, X)
        below = torch.cat([zero, L[:, :Z - 1]], dim=1)
        above = torch.cat([L[:, 1:], zero], dim=1)
        col = obs.unsqueeze(1).expand(B, Z, C, Y, X)
        zid = self.zid.to(obs.dtype).reshape(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        u = torch.cat([L, below, above, col, zid], dim=2).reshape(B * Z, -1, Y, X)
        e = self.cell(u)
        p = F.relu(self.proj(e))
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                z = p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                z = torch.cat([q[:, :, oy:oy + Y, ox:ox + X] for oy in range(dy) for ox in range(dx)], dim=1)
            res.append(out(F.relu(mix(z))))
        T = len(self.fps)
        s = torch.cat(res, dim=1).reshape(B, Z, T, Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        ev = e.reshape(B, Z, E, Y, X)
        g = self.glob(torch.cat([ev.mean((1, 3, 4)), ev.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
