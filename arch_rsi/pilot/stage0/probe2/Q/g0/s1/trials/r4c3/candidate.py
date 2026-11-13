import torch
import torch.nn as nn
import torch.nn.functional as F

W = 24
HID = 12
CW = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Conv3d(5, W, 3, padding=1)
        self.plane = nn.Conv2d(spec.in_channels - 4 * self.Z, W, 1)
        self.conv_xy = nn.Conv3d(W, W, (1, 3, 3), padding=(0, 1, 1))
        self.conv_dil = nn.Conv3d(W, W, (1, 3, 3), padding=(0, 2, 2), dilation=(1, 2, 2))
        self.col_mix = nn.Conv3d(3 * W, W, 1)
        self.heads = nn.ModuleList([nn.Conv3d(W, HID, (2, dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv3d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.critic = nn.Sequential(
            nn.Conv2d(spec.in_channels, CW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(CW, CW, 3, padding=1), nn.ReLU())
        self.v_mlp = nn.Sequential(nn.Linear(2 * CW, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, obs, mask):
        B = obs.shape[0]
        Z, Y, X = self.Z, self.Y, self.X
        vox = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        zc = torch.arange(Z, dtype=obs.dtype, device=obs.device) / Z
        zc = zc.view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        vox = torch.cat([vox, zc], dim=1)
        f = F.relu(self.stem(vox) + self.plane(obs[:, 4 * Z:]).unsqueeze(2))
        f = f + F.relu(self.conv_xy(f))
        f = f + F.relu(self.conv_dil(f))
        up = torch.cummax(f, dim=2).values
        down = torch.flip(torch.cummax(torch.flip(f, [2]), dim=2).values, [2])
        f = f + F.relu(self.col_mix(torch.cat([f, up, down], dim=1)))
        res = []
        for fp, conv, out in zip(self.fps, self.heads, self.outs):
            dy, dx = fp
            h = F.pad(f, (0, dx - 1, 0, dy - 1, 1, 0))
            s = out(F.relu(conv(h)))
            res.append(torch.logsumexp(s, dim=2))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3, 4)), f.amax((2, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        c = self.critic(obs)
        value = self.v_mlp(torch.cat([c.mean((2, 3)), c.amax((2, 3))], dim=1)).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
