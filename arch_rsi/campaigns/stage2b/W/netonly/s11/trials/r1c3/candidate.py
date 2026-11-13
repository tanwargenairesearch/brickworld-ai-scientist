import torch
import torch.nn as nn
import torch.nn.functional as F

LW = 16
WIDTH = 48
N_LAYER_IN = 12


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.layer_enc = nn.Sequential(
            nn.Conv2d(N_LAYER_IN, LW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(LW, LW, 3, padding=1), nn.ReLU())
        self.merge = nn.Conv2d(LW * spec.Z, WIDTH, 1)
        self.spatial = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.place_heads = nn.ModuleList([nn.Conv2d(WIDTH, 1, (dy, dx)) for dy, dx in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        Z = self.Z
        B, _, Y, X = obs.shape
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        b_below = F.pad(b[:, :Z - 1], (0, 0, 0, 0, 1, 0), value=1.0)
        t_below = F.pad(t[:, :Z - 1], (0, 0, 0, 0, 1, 0), value=1.0)
        b_above = F.pad(b[:, 1:], (0, 0, 0, 0, 0, 1), value=0.0)
        t_above = F.pad(t[:, 1:], (0, 0, 0, 0, 0, 1), value=0.0)
        per = torch.stack([b, t, m, e, b_below, t_below, b_above, t_above], dim=2)
        glob_ch = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([per, glob_ch, zc], dim=2).reshape(B * Z, N_LAYER_IN, Y, X)
        h = self.layer_enc(x).reshape(B, Z * LW, Y, X)
        h = F.relu(self.merge(h))
        f = F.relu(self.spatial(h))
        outs = []
        for (dy, dx), head in zip(self.fps, self.place_heads):
            outs.append(head(F.pad(f, (0, dx - 1, 0, dy - 1))))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
