import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 40
LAND_HID = 16
D = 12
H = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.n_extra = int(spec.in_channels) - 4 * Z
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.land_hid = nn.ModuleList([nn.Conv2d(WIDTH, LAND_HID, (dy, dx)) for dy, dx in self.fps])
        self.land_out = nn.ModuleList([nn.Conv2d(LAND_HID, Z, 1) for _ in self.fps])
        self.bias = nn.ModuleList([nn.Conv2d(WIDTH, 1, (dy, dx)) for dy, dx in self.fps])
        self.cell = nn.Conv2d(4 + Z + self.n_extra, D, 1)
        self.score_hid = nn.ModuleList([nn.Conv2d(2 * D, H, (dy, dx)) for dy, dx in self.fps])
        self.score_out = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        for m in self.score_out:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        for m in self.bias:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        f = self.body(obs)
        per = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 2, 1, 3, 4)
        zid = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        ext = obs[:, 4 * Z:].unsqueeze(1).expand(B, Z, self.n_extra, Y, X)
        x = torch.cat([per, zid, ext], dim=2).reshape(B * Z, 4 + Z + self.n_extra, Y, X)
        u = F.relu(self.cell(x)).view(B, Z, D, Y, X)
        below = torch.cumsum(u, dim=1) - u
        v = torch.cat([u, below], dim=2).reshape(B * Z, 2 * D, Y, X)
        outs = []
        for i, (dy, dx) in enumerate(self.fps):
            pad = (0, dx - 1, 0, dy - 1)
            fp = F.pad(f, pad)
            p = torch.softmax(self.land_out[i](F.relu(self.land_hid[i](fp))), dim=1)
            h = F.relu(self.score_hid[i](F.pad(v, pad)))
            s = self.score_out[i](h).view(B, Z, Y, X)
            logit = (p * s).sum(1) + self.bias[i](fp).squeeze(1)
            outs.append(logit.unsqueeze(1))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
