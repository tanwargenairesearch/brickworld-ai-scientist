import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 17
HID = 64
BLOCKS = 2


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        c_in = 4 * spec.Z + 4
        self.stem = nn.Conv2d(c_in, WIDTH, 3, padding=1)
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(BLOCKS)])
        self.mix = nn.Conv2d(WIDTH, WIDTH, 1)
        self.foot = [(int(f[0]), int(f[1])) for f in spec.footprints]
        self.heads = nn.ModuleList([nn.Conv2d(WIDTH, 1, (dy, dx)) for dy, dx in self.foot])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, HID), nn.ReLU())
        self.done_head = nn.Linear(HID, 1)
        self.value_head = nn.Linear(HID, 1)

    def forward(self, obs, mask):
        Z = self.Z
        B, C, Y, X = obs.shape
        kinds = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        padded = F.pad(kinds, (0, 0, 0, 0, Z - 1, 0))
        rel = torch.stack([padded[:, :, z:z + Z] for z in range(Z)], dim=1)
        rel = rel.reshape(B, Z, 4 * Z, Y, X)
        extra = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([rel, extra, zc], dim=2).reshape(B * Z, 4 * Z + 4, Y, X)
        h = F.relu(self.stem(x))
        for conv in self.res:
            h = F.relu(h + conv(h))
        h = F.relu(self.mix(h))
        outs = []
        for (dy, dx), head in zip(self.foot, self.heads):
            outs.append(head(F.pad(h, (0, dx - 1, 0, dy - 1))))
        per_layer = torch.cat(outs, dim=1).reshape(B, Z, len(self.foot), Y, X)
        place = torch.logsumexp(per_layer, dim=1).flatten(1)
        hz = h.reshape(B, Z, -1, Y, X)
        g = self.glob(torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
