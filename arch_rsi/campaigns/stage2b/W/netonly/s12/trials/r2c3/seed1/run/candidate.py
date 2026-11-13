import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 56
K = 8
KL = 4


def shifts(x, dy, dx, ny, nx):
    xp = F.pad(x, (0, dx - 1, 0, dy - 1))
    return torch.stack([xp[:, :, i:i + ny, j:j + nx] for i in range(dy) for j in range(dx)], 0)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        nt = spec.n_types
        z = spec.Z
        self.nt = nt
        self.z = z
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.body = nn.Sequential(
            nn.Conv2d(c, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.cell = nn.Conv2d(WIDTH, nt * K, 1)
        self.lay = nn.Conv2d(WIDTH, nt * z * KL, 1)
        self.hgt = nn.Conv2d(WIDTH, nt * 2 * z, 1)
        self.mix = nn.Conv2d(nt * 3 * K, nt, 1, groups=nt)
        self.land = nn.Conv2d(nt * 2 * z, nt * z, 1, groups=nt)
        self.score = nn.Conv2d(nt * z * 2 * KL, nt * z, 1, groups=nt * z)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        b, _, ny, nx = f.shape
        nt, z = self.nt, self.z
        h = self.cell(f).view(b, nt, K, ny, nx)
        la = self.lay(f).view(b, nt, z * KL, ny, nx)
        hg = self.hgt(f).view(b, nt, 2 * z, ny, nx)
        outs, louts, souts = [], [], []
        for t, (dy, dx) in enumerate(self.fps):
            st = shifts(h[:, t], dy, dx, ny, nx)
            outs += [st.amin(0), st.amax(0), st.mean(0)]
            sl = shifts(la[:, t], dy, dx, ny, nx)
            lmin = sl.amin(0).view(b, z, KL, ny, nx)
            lmean = sl.mean(0).view(b, z, KL, ny, nx)
            louts.append(torch.cat([lmin, lmean], 2).reshape(b, z * 2 * KL, ny, nx))
            sg = shifts(hg[:, t], dy, dx, ny, nx)
            souts += [sg[:, :, :z].amax(0), sg[:, :, z:].amin(0)]
        place1 = self.mix(torch.cat(outs, dim=1))
        w = torch.softmax(self.land(torch.cat(souts, dim=1)).view(b, nt, z, ny, nx), dim=2)
        sc = self.score(torch.cat(louts, dim=1)).view(b, nt, z, ny, nx)
        place = (place1 + (w * sc).sum(2)).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
