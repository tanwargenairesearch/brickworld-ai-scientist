import torch
import torch.nn as nn
import torch.nn.functional as F

W3 = 12
GW = 64
HID = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = spec.Z
        self.Z = Z
        self.c3a = nn.Conv3d(10, W3, 3, padding=1)
        self.g3 = nn.Linear(2 * W3 * Z, W3 * Z)
        self.c3b = nn.Conv3d(W3, W3, 3, padding=1)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.heads = nn.ModuleList([nn.Conv3d(W3, HID, (1, dy, dx)) for (dy, dx) in self.fps])
        self.outs = nn.ModuleList([nn.Conv3d(HID, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W3 * Z, GW), nn.ReLU())
        self.done_head = nn.Linear(GW, 1)
        self.value_head = nn.Linear(GW, 1)

    def pool(self, v):
        return torch.cat([v.mean((3, 4)).flatten(1), v.amax((3, 4)).flatten(1)], dim=1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        vol = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        sc = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = torch.linspace(-1.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        yc = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, 1, Y, 1).expand(B, 1, Z, Y, X)
        xc = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, 1, X).expand(B, 1, Z, Y, X)
        v = F.relu(self.c3a(torch.cat([vol, sc, zc, yc, xc], dim=1)))
        gc = self.g3(self.pool(v)).reshape(B, W3, Z, 1, 1)
        v = v + F.relu(self.c3b(v) + gc)
        outs = []
        for (dy, dx), hd, o in zip(self.fps, self.heads, self.outs):
            p = F.pad(v, (0, dx - 1, 0, dy - 1))
            q = o(F.relu(hd(p)))
            a = F.log_softmax(q[:, 0], dim=1)
            outs.append(torch.logsumexp(a + q[:, 1], dim=1))
        place = torch.stack(outs, dim=1).flatten(1)
        g = self.glob(self.pool(v))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
