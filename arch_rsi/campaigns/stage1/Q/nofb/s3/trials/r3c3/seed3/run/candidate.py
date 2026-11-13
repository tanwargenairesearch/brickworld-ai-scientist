import torch
import torch.nn as nn
import torch.nn.functional as F

W = 24
HW = 16
GW = 64
W2 = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.fps = [(int(a), int(b)) for (a, b) in spec.footprints]
        self.c1 = nn.Conv2d(10, W, 3, padding=1)
        self.gproj = nn.Conv2d(spec.in_channels, W, 1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(W, HW, (dy, dx)), nn.ReLU(), nn.Conv2d(HW, 1, 1))
            for (dy, dx) in self.fps])
        for hd in self.heads:
            nn.init.zeros_(hd[2].weight)
            nn.init.zeros_(hd[2].bias)
        self.body2 = nn.Sequential(nn.Conv2d(spec.in_channels, W2, 3, padding=1), nn.ReLU())
        self.head2 = nn.Conv2d(W2, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W + 2 * W2, GW), nn.ReLU())
        self.done_head = nn.Linear(GW, 1)
        self.value_head = nn.Linear(GW, 1)

    def forward(self, obs, mask):
        Z = self.Z
        B, C, Y, X = obs.shape
        T = len(self.fps)
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        below = F.pad(b, (0, 0, 0, 0, 1, 0))[:, :Z]
        tab = F.pad(t, (0, 0, 0, 0, 0, 1))[:, 1:]
        per = torch.stack([b, t, m, e, below, tab], dim=2)
        bc = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1)
        zc = zc.view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        v = torch.cat([per, bc, zc], dim=2).reshape(B * Z, 10, Y, X)
        g0 = self.gproj(obs).unsqueeze(1).expand(B, Z, W, Y, X).reshape(B * Z, W, Y, X)
        h = F.relu(self.c1(v) + g0)
        h = h + F.relu(self.c2(h))
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            q = F.pad(h, (0, dx - 1, 0, dy - 1))
            outs.append(head(q))
        s = torch.cat(outs, dim=1).reshape(B, Z, T, Y, X)
        f2 = self.body2(obs)
        place = (torch.logsumexp(s, dim=1) + self.head2(f2)).flatten(1)
        hv = h.reshape(B, Z, W, Y, X)
        g = self.glob(torch.cat([hv.mean((1, 3, 4)), hv.amax((1, 3, 4)),
                                 f2.mean((2, 3)), f2.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
