import torch
import torch.nn as nn
import torch.nn.functional as F

LW = 24
WIDTH = 64
N_IN = 10


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.layer_enc = nn.Sequential(
            nn.Conv2d(N_IN, LW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(LW, LW, 3, padding=1), nn.ReLU())
        self.mix = nn.Sequential(nn.Conv2d(2 * LW, WIDTH, 3, padding=1), nn.ReLU())
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        build = obs[:, 0:Z]
        target = obs[:, Z:2 * Z]
        missing = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        build_below = F.pad(build, (0, 0, 0, 0, 1, 0))[:, :Z]
        target_below = F.pad(target, (0, 0, 0, 0, 1, 0))[:, :Z]
        per = torch.stack([build, target, missing, extra, build_below, target_below], dim=2)
        gch = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = (torch.arange(Z, dtype=obs.dtype, device=obs.device) / max(Z - 1, 1)).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([per, gch, zc], dim=2).reshape(B * Z, N_IN, Y, X)
        h = self.layer_enc(x).reshape(B, Z, LW, Y, X)
        f = torch.cat([h.amax(1), h.mean(1)], dim=1)
        f = self.mix(f)
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
