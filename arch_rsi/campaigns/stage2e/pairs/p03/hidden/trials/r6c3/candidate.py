import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.98}

FW = 48
FPW = 24
FHID = 24
LW = 16
LPW = 12
LHID = 12
GW = 64


def fpool(q, dy, dx):
    qp = F.pad(q, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(qp, (dy, dx), stride=1)
    mn = -F.max_pool2d(-qp, (dy, dx), stride=1)
    av = F.avg_pool2d(qp, (dy, dx), stride=1)
    return torch.cat([mx, mn, av], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        c = spec.in_channels
        self.fbody = nn.Sequential(
            nn.Conv2d(c, FW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(FW, FW, 3, padding=1), nn.ReLU())
        self.fproj = nn.Sequential(nn.Conv2d(FW, FPW, 1), nn.ReLU())
        self.fheads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * FPW + FW, FHID, 1), nn.ReLU(), nn.Conv2d(FHID, 1, 1))
            for _ in self.fps])
        n_in = 12 + 3 + 1
        self.lbody = nn.Sequential(
            nn.Conv2d(n_in, LW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(LW, LW, 3, padding=1), nn.ReLU())
        self.lproj = nn.Sequential(nn.Conv2d(LW, LPW, 1), nn.ReLU())
        self.lheads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * LPW, LHID, 1), nn.ReLU(), nn.Conv2d(LHID, 1, 1))
            for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * FW + 4 * LW, GW), nn.ReLU())
        self.done_head = nn.Linear(GW, 1)
        self.value_head = nn.Linear(GW, 1)

    def forward(self, obs, mask):
        b, _, yy, xx = obs.shape
        Z = self.Z
        T = len(self.fps)
        f = self.fbody(obs)
        q = self.fproj(f)
        fouts = []
        for (dy, dx), head in zip(self.fps, self.fheads):
            fouts.append(head(torch.cat([fpool(q, dy, dx), f], dim=1)))
        fplace = torch.cat(fouts, dim=1)
        g = obs[:, :4 * Z].reshape(b, 4, Z, yy, xx)
        gp = F.pad(g, (0, 0, 0, 0, 1, 1))
        sl = torch.stack([gp[:, :, z:z + 3] for z in range(Z)], dim=1)
        sl = sl.reshape(b, Z, 12, yy, xx)
        ext = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(b, Z, 3, yy, xx)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype)
        zc = zc.view(1, Z, 1, 1, 1).expand(b, Z, 1, yy, xx)
        x = torch.cat([sl, ext, zc], dim=2).reshape(b * Z, 16, yy, xx)
        h = self.lbody(x)
        lq = self.lproj(h)
        louts = []
        for (dy, dx), head in zip(self.fps, self.lheads):
            louts.append(head(fpool(lq, dy, dx)))
        pl = torch.cat(louts, dim=1).reshape(b, Z, T, yy, xx)
        lplace = torch.logsumexp(pl, dim=1)
        place = (fplace + lplace).flatten(1)
        hz = h.reshape(b, Z, LW, yy, xx)
        cell = torch.cat([hz.mean(1), hz.amax(1)], dim=1)
        gin = torch.cat([f.mean((2, 3)), f.amax((2, 3)), cell.mean((2, 3)), cell.amax((2, 3))], dim=1)
        gf = self.glob(gin)
        logits = torch.cat([place, self.done_head(gf)], dim=1)
        return logits, self.value_head(gf).squeeze(1)


def build_policy(spec):
    return Net(spec)
