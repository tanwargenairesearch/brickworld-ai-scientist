import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 32
N_BLOCKS = 4
HID = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.n_types = spec.n_types
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(N_BLOCKS)])
        self.norms = nn.ModuleList([nn.LayerNorm(WIDTH) for _ in range(N_BLOCKS)])
        self.final_norm = nn.LayerNorm(WIDTH)
        self.type_feat = nn.Conv2d(WIDTH, self.n_types * HID, 1)
        self.type_out = nn.ModuleList([nn.Conv2d(3 * HID, 1, 1) for _ in range(self.n_types)])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def cell_norm(self, norm, f):
        return norm(f.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, obs, mask):
        f = self.stem(obs)
        for conv, norm in zip(self.blocks, self.norms):
            f = f + conv(F.relu(self.cell_norm(norm, f)))
        f = F.relu(self.cell_norm(self.final_norm, f))
        h = self.type_feat(f)
        outs = []
        for t in range(self.n_types):
            dy, dx = self.footprints[t]
            ht = h[:, t * HID:(t + 1) * HID]
            hp = F.pad(ht, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(hp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-hp, (dy, dx), stride=1)
            mean = F.avg_pool2d(hp, (dy, dx), stride=1)
            outs.append(self.type_out[t](torch.cat([mx, mn, mean], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
