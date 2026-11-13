import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
GW = 64
D = 24
HID = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GW), nn.ReLU())
        self.ctx = nn.Linear(GW, D)
        self.cproj = nn.Conv2d(WIDTH, D, 1)
        self.lay1 = nn.Conv3d(4 + 3 + Z + 1, D, 1)
        self.lay2 = nn.Conv3d(D, D, 1)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv3d(2 * D, HID, 1), nn.ReLU(), nn.Conv3d(HID, 1, 1))
            for _ in self.fps])
        self.place2d = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.done_head = nn.Linear(GW, 1)
        self.value_head = nn.Linear(GW, 1)
        if Z > 1:
            zlin = torch.linspace(0.0, 1.0, Z)
        else:
            zlin = torch.zeros(1)
        zc = torch.cat([torch.eye(Z), zlin.view(1, Z)], dim=0)
        self.register_buffer('zc', zc)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        vox = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        planes = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = self.zc.view(1, Z + 1, Z, 1, 1).expand(B, Z + 1, Z, Y, X)
        x = torch.cat([vox, planes, zc], dim=1)
        f = self.body(obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        c2 = self.cproj(f) + self.ctx(g)[:, :, None, None]
        u = F.relu(self.lay1(x) + c2.unsqueeze(2))
        u = F.relu(self.lay2(u))
        outs = []
        for fp, head in zip(self.fps, self.heads):
            dy, dx = fp
            if dy == 1 and dx == 1:
                mx = u
                mn = u
            else:
                p = F.pad(u, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool3d(p, (1, dy, dx), stride=1)
                mn = -F.max_pool3d(-p, (1, dy, dx), stride=1)
            s = head(torch.cat([mx, mn], dim=1))
            outs.append(torch.logsumexp(s, dim=2))
        place = (torch.cat(outs, dim=1) + self.place2d(f)).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
