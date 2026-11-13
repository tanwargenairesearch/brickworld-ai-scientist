import torch
import torch.nn as nn
import torch.nn.functional as F

CH = 20
GW = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.body = nn.Sequential(
            nn.Conv3d(8, CH, 3, padding=1), nn.ReLU(),
            nn.Conv3d(CH, CH, 3, padding=1), nn.ReLU())
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.heads = nn.ModuleList([nn.Conv3d(CH, 2, (1, dy, dx)) for (dy, dx) in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * CH, GW), nn.ReLU())
        self.done_head = nn.Linear(GW, 1)
        self.value_head = nn.Linear(GW, 1)
        self.register_buffer('zc', torch.linspace(0.0, 1.0, spec.Z).view(1, 1, spec.Z, 1, 1))

    def forward(self, obs, mask):
        B, Y, X = obs.shape[0], obs.shape[2], obs.shape[3]
        Z = self.Z
        vol = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        planes = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = self.zc.expand(B, 1, Z, Y, X)
        h = self.body(torch.cat([vol, planes, zc], dim=1))
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            hp = F.pad(h, (0, dx - 1, 0, dy - 1, 0, 0))
            o = head(hp)
            s = o[:, 0]
            w = torch.softmax(o[:, 1], dim=1)
            outs.append((w * s).sum(1))
        place = torch.stack(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3, 4)), h.amax((2, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
