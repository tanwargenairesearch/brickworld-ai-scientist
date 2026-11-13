import torch
import torch.nn as nn
import torch.nn.functional as F

BW = 32
E = 32
H = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, BW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(BW, BW, 3, padding=1), nn.ReLU())
        self.layer_in = nn.Conv2d(9 + self.Z, E, 1)
        self.ctx = nn.Conv2d(BW, E, 1)
        self.layer_mid = nn.Conv2d(E, E, 1)
        self.type_hidden = nn.ModuleList([nn.Conv2d(2 * E, H, 1) for _ in self.fps])
        self.type_out = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * BW + 2 * E, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        f = self.body(obs)
        build = obs[:, 0:Z]
        target = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        bb = F.pad(build, (0, 0, 0, 0, 1, 0))[:, :Z]
        tb = F.pad(target, (0, 0, 0, 0, 1, 0))[:, :Z]
        per = torch.stack([build, target, miss, extra, bb, tb], dim=2)
        gc = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zp = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        p = torch.cat([per, gc, zp], dim=2).reshape(B * Z, 9 + Z, Y, X)
        c = self.ctx(f).unsqueeze(1).expand(B, Z, E, Y, X).reshape(B * Z, E, Y, X)
        e = F.relu(self.layer_in(p) + c)
        e = F.relu(self.layer_mid(e))
        outs = []
        for i, (dy, dx) in enumerate(self.fps):
            ep = F.pad(e, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(ep, (dy, dx), stride=1)
            mn = -F.max_pool2d(-ep, (dy, dx), stride=1)
            h = F.relu(self.type_hidden[i](torch.cat([mx, mn], dim=1)))
            s = self.type_out[i](h).view(B, Z, Y, X)
            outs.append(torch.logsumexp(s, dim=1))
        place = torch.stack(outs, dim=1).flatten(1)
        e5 = e.view(B, Z, E, Y, X)
        gl = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)),
                                  e5.mean((1, 3, 4)), e5.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(gl)], dim=1)
        return logits, self.value_head(gl).squeeze(1)


def build_policy(spec):
    return Net(spec)
