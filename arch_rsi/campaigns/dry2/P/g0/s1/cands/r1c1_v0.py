import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
HID = 32


def fp_pool(x, dy, dx, mode):
    if dy > 1 or dx > 1:
        x = F.pad(x, (0, dx - 1, 0, dy - 1))
    if mode == 'max':
        return F.max_pool2d(x, (dy, dx), stride=1)
    return F.avg_pool2d(x, (dy, dx), stride=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = spec.Z
        self.Z = Z
        self.T = spec.n_types
        self.fps = [(int(a), int(b)) for (a, b) in spec.footprints]
        C = spec.in_channels
        self.body = nn.Sequential(
            nn.Conv2d(C, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        nin = WIDTH + (Z + 6) + self.T
        self.head1 = nn.Conv2d(nin, HID, 1)
        self.head2 = nn.Conv2d(HID, 1, 1)
        with torch.no_grad():
            self.head2.weight.mul_(0.1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + C, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def hand(self, obs, dy, dx):
        Z = self.Z
        hb = torch.round(obs[:, 4 * Z:4 * Z + 1] * Z)
        ht = torch.round(obs[:, 4 * Z + 1:4 * Z + 2] * Z)
        L = fp_pool(hb, dy, dx, 'max')
        zs1 = torch.arange(Z + 1, device=obs.device, dtype=obs.dtype).view(1, -1, 1, 1)
        oh = (L == zs1).to(obs.dtype)
        tavg = fp_pool(obs[:, Z:2 * Z], dy, dx, 'avg')
        mavg = fp_pool(obs[:, 2 * Z:3 * Z], dy, dx, 'avg')
        tL = (oh[:, :Z] * tavg).sum(1, keepdim=True)
        tA = (oh[:, :Z - 1] * tavg[:, 1:]).sum(1, keepdim=True)
        below = (L > zs1[:, :Z]).to(obs.dtype)
        gm = (below * mavg).sum(1, keepdim=True)
        flat = (L - fp_pool(hb, dy, dx, 'avg')) / Z
        tg = (fp_pool(ht, dy, dx, 'avg') - L) / Z
        return torch.cat([oh, tL, tA, gm, flat, tg], 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        f = self.body(obs)
        feats = []
        for t, (dy, dx) in enumerate(self.fps):
            ft = fp_pool(f, dy, dx, 'avg')
            h = self.hand(obs, dy, dx)
            to = obs.new_zeros(B, self.T, Y, X)
            to[:, t] = 1.0
            feats.append(torch.cat([ft, h, to], 1))
        x = torch.cat(feats, 0)
        p = self.head2(F.relu(self.head1(x)))
        place = p.reshape(self.T, B, Y * X).permute(1, 0, 2).reshape(B, -1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), obs.mean((2, 3))], 1))
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
