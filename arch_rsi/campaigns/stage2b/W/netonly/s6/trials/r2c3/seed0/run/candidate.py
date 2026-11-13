import torch
import torch.nn as nn
import torch.nn.functional as F

LW = 16
WIDTH = 32
BLOCKS = 2


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        Z = spec.Z
        self.register_buffer('zpos', (torch.arange(Z, dtype=torch.float32) / Z).view(1, Z, 1, 1, 1))
        self.layer_enc = nn.Sequential(
            nn.Conv2d(12, LW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(LW, LW, 3, padding=1), nn.ReLU())
        self.mix = nn.Conv2d(Z * LW, WIDTH, 1)
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(BLOCKS)])
        self.mid_glob = nn.Linear(2 * WIDTH, WIDTH)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        build = obs[:, 0:Z]
        target = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        rest = obs[:, 4 * Z:4 * Z + 3]
        bp = F.pad(build, (0, 0, 0, 0, 1, 1))
        tp = F.pad(target, (0, 0, 0, 0, 1, 1))
        per = torch.stack([build, target, miss, extra,
                           bp[:, 0:Z], tp[:, 0:Z], bp[:, 2:Z + 2], tp[:, 2:Z + 2]], dim=2)
        rest_e = rest[:, None].expand(B, Z, 3, Y, X)
        zp = self.zpos.expand(B, Z, 1, Y, X)
        inp = torch.cat([per, rest_e, zp], dim=2).reshape(B * Z, 12, Y, X)
        h = self.layer_enc(inp).reshape(B, Z * LW, Y, X)
        x = self.mix(h)
        for i, conv in enumerate(self.blocks):
            x = x + conv(F.relu(x))
            if i == 0:
                r = F.relu(x)
                g0 = self.mid_glob(torch.cat([r.mean((2, 3)), r.amax((2, 3))], dim=1))
                x = x + g0[:, :, None, None]
        f = F.relu(x)
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
