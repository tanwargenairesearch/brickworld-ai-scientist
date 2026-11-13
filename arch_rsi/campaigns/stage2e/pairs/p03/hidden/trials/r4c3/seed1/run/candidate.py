import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "gamma": 0.98}

LW = 24
PW = 16
HID = 16
GW = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        n_in = 12 + 3 + 1
        self.c1 = nn.Conv2d(n_in, LW, 3, padding=1)
        self.z1 = nn.Conv2d(LW, LW, (3, 1), padding=(1, 0))
        self.c2 = nn.Conv2d(LW, LW, 3, padding=1)
        self.z2 = nn.Conv2d(LW, LW, (3, 1), padding=(1, 0))
        self.proj = nn.Sequential(nn.Conv2d(LW, PW, 1), nn.ReLU())
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * PW, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(4 * LW, GW), nn.ReLU())
        self.done_head = nn.Linear(GW, 1)
        self.value_head = nn.Linear(GW, 1)

    def zmix(self, h, conv, b, yy, xx):
        Z = self.Z
        c = h.shape[1]
        t = h.reshape(b, Z, c, yy * xx).transpose(1, 2)
        t = F.relu(conv(t))
        return t.transpose(1, 2).reshape(b * Z, c, yy, xx)

    def forward(self, obs, mask):
        b, _, yy, xx = obs.shape
        Z = self.Z
        g = obs[:, :4 * Z].reshape(b, 4, Z, yy, xx)
        gp = F.pad(g, (0, 0, 0, 0, 1, 1))
        sl = torch.stack([gp[:, :, z:z + 3] for z in range(Z)], dim=1)
        sl = sl.reshape(b, Z, 12, yy, xx)
        ext = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(b, Z, 3, yy, xx)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype)
        zc = zc.view(1, Z, 1, 1, 1).expand(b, Z, 1, yy, xx)
        x = torch.cat([sl, ext, zc], dim=2).reshape(b * Z, 16, yy, xx)
        h = F.relu(self.c1(x))
        h = h + self.zmix(h, self.z1, b, yy, xx)
        h = h + F.relu(self.c2(h))
        f = h + self.zmix(h, self.z2, b, yy, xx)
        q = self.proj(f)
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            qp = F.pad(q, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(qp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-qp, (dy, dx), stride=1)
            av = F.avg_pool2d(qp, (dy, dx), stride=1)
            outs.append(head(torch.cat([mx, mn, av], dim=1)))
        pl = torch.cat(outs, dim=1).reshape(b, Z, len(self.fps), yy, xx)
        place = torch.logsumexp(pl, dim=1).flatten(1)
        fz = f.reshape(b, Z, LW, yy, xx)
        cell = torch.cat([fz.mean(1), fz.amax(1)], dim=1)
        gf = self.glob(torch.cat([cell.mean((2, 3)), cell.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(gf)], dim=1)
        return logits, self.value_head(gf).squeeze(1)


def build_policy(spec):
    return Net(spec)
