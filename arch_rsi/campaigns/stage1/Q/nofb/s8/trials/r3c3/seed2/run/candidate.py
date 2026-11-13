import torch
import torch.nn as nn
import torch.nn.functional as F

W1 = 48
WIDTH = 64
H = 4
D = 8


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Conv2d(spec.in_channels, W1, 3, padding=1)
        self.q = nn.Conv2d(spec.in_channels, H * self.Z, 3, padding=1)
        self.v = nn.Conv3d(5, H * D, (1, 3, 3), padding=(0, 1, 1))
        self.mix = nn.Conv2d(W1 + H * D, WIDTH, 3, padding=1)
        self.heads = nn.ModuleList([nn.Conv2d(4 * WIDTH, 1, 1) for _ in self.footprints])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        vox = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        zc = torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1)
        zc = zc.view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        v = self.v(torch.cat([vox, zc], dim=1)).view(B, H, D, Z, Y, X)
        a = torch.softmax(self.q(obs).view(B, H, Z, Y, X), dim=2).unsqueeze(2)
        col = (a * v).sum(3).reshape(B, H * D, Y, X)
        h = F.relu(self.stem(obs))
        f = F.relu(self.mix(torch.cat([h, col], dim=1)))
        outs = []
        for (dy, dx), head in zip(self.footprints, self.heads):
            fp = F.pad(f, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(fp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-fp, (dy, dx), stride=1)
            av = F.avg_pool2d(fp, (dy, dx), stride=1)
            outs.append(head(torch.cat([f, mx, mn, av], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
