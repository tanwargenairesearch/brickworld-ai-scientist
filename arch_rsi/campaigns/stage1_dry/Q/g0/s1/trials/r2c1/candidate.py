import torch
import torch.nn as nn
import torch.nn.functional as F

W2 = 32
WV = 32
WP = 16
WH = 16
WG = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.fps = [(int(f[0]), int(f[1])) for f in spec.footprints]
        self.trunk = nn.Sequential(
            nn.Conv2d(spec.in_channels, W2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(W2, W2, 3, padding=1), nn.ReLU())
        vin = 7 + 3 + 1 + W2
        self.vox1 = nn.Sequential(nn.Conv3d(vin, WV, 1), nn.ReLU())
        self.vox2 = nn.Sequential(nn.Conv3d(3 * WV, WP, 1), nn.ReLU())
        self.heads = nn.ModuleList()
        for _ in self.fps:
            last = nn.Conv3d(WH, 1, 1)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            self.heads.append(nn.Sequential(nn.Conv3d(3 * WP, WH, 1), nn.ReLU(), last))
        self.glob = nn.Sequential(nn.Linear(2 * W2 + 2 * WP, WG), nn.ReLU())
        self.done_head = nn.Linear(WG, 1)
        self.value_head = nn.Linear(WG, 1)

    def forward(self, obs, mask):
        Z = self.Z
        B, _, Y, X = obs.shape
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        b_below = F.pad(b, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        t_below = F.pad(t, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        b_above = F.pad(b, (0, 0, 0, 0, 0, 1))[:, 1:]
        layers = torch.stack([b, t, m, e, b_below, t_below, b_above], dim=1)
        planes = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zpos = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        f2 = self.trunk(obs)
        f2v = f2.unsqueeze(2).expand(B, f2.shape[1], Z, Y, X)
        v1 = self.vox1(torch.cat([layers, planes, zpos, f2v], dim=1))
        below = torch.cummax(v1, dim=2).values
        above = torch.flip(torch.cummax(torch.flip(v1, [2]), dim=2).values, [2])
        v = self.vox2(torch.cat([v1, below, above], dim=1))
        vf = v.reshape(B, WP * Z, Y, X)
        outs = []
        for fp, head in zip(self.fps, self.heads):
            dy, dx = fp
            p = F.pad(vf, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(p, (dy, dx), stride=1).reshape(B, WP, Z, Y, X)
            mn = (-F.max_pool2d(-p, (dy, dx), stride=1)).reshape(B, WP, Z, Y, X)
            s = head(torch.cat([v, mx, mn], dim=1)).squeeze(1)
            outs.append(torch.logsumexp(s, dim=1))
        place = torch.stack(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f2.mean((2, 3)), f2.amax((2, 3)),
                                 v.mean((2, 3, 4)), v.amax((2, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
