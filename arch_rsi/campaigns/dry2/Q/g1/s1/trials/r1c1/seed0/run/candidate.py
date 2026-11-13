import torch
import torch.nn as nn
import torch.nn.functional as F

GAIN = 15.0


def wsum(x, dy, dx):
    xp = F.pad(x, (0, dx - 1, 0, dy - 1))
    return F.avg_pool2d(xp, (dy, dx), stride=1) * float(dy * dx)


def wmax(x, dy, dx):
    xp = F.pad(x, (0, dx - 1, 0, dy - 1))
    return F.max_pool2d(xp, (dy, dx), stride=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.T = spec.n_types
        self.n_place = spec.n_place
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.amax = float(max(a * b for a, b in self.fps))
        self.place_w = nn.Linear(7, 1, bias=False)
        nn.init.zeros_(self.place_w.weight)
        self.type_b = nn.Parameter(torch.zeros(self.T))
        self.done_w = nn.Linear(4, 1, bias=False)
        nn.init.zeros_(self.done_w.weight)
        self.value = nn.Sequential(nn.Linear(6, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, obs, mask):
        Z = self.Z
        B = obs.shape[0]
        target = obs[:, Z:2 * Z]
        missing = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        hb = torch.round(obs[:, 4 * Z:4 * Z + 1] * Z)
        tfrac = obs[:, 4 * Z + 2].mean((1, 2))
        ar = torch.arange(Z, device=obs.device, dtype=obs.dtype).view(1, Z, 1, 1)
        mg = missing * (ar >= hb).to(obs.dtype)
        unsup_t = target * (ar > hb).to(obs.dtype)
        lay_any = (mg.amax((2, 3)) > 0.5).to(obs.dtype)
        m = (ar.view(1, Z) + Z * (1.0 - lay_any)).amin(1).view(B, 1, 1)
        feats, cleans = [], []
        for (dy, dx) in self.fps:
            area = float(dy * dx)
            L = wmax(hb, dy, dx)
            oh = (ar == L).to(obs.dtype)
            below = (ar < L).to(obs.dtype)
            mS = wsum(mg, dy, dx)
            good = (mS * oh).sum(1)
            buried = (mS * below).sum(1)
            unsup = (wsum(unsup_t, dy, dx) * oh).sum(1)
            ext = area - good
            Lf = L[:, 0] / Z
            rel = (L[:, 0] - m) / Z
            ar_f = torch.full_like(good, area / self.amax)
            f = torch.stack([good / self.amax, ext / self.amax, buried / self.amax,
                             unsup / self.amax, Lf, rel, ar_f], dim=-1)
            feats.append(f)
            cleans.append(good / self.amax * (ext < 0.5).to(obs.dtype))
        feats = torch.stack(feats, dim=1)
        place = self.place_w(feats).squeeze(-1) * GAIN + self.type_b.view(1, -1, 1, 1) * GAIN
        place = place.flatten(1)
        legal = mask[:, :self.n_place].to(obs.dtype)
        best = (torch.stack(cleans, dim=1).flatten(1) * legal).amax(1)
        tot = target.sum((1, 2, 3)).clamp(min=1.0)
        msum = missing.sum((1, 2, 3))
        esum = extra.sum((1, 2, 3))
        mfrac = msum / tot
        efrac = esum / tot
        exact = ((msum + esum) < 0.5).to(obs.dtype)
        sealed = (missing - mg).sum((1, 2, 3)) / tot
        dfe = torch.stack([mfrac, efrac, best, exact], dim=1)
        done = self.done_w(dfe) * GAIN
        logits = torch.cat([place, done], dim=1)
        vfe = torch.stack([mfrac, efrac, best, exact, tfrac, sealed], dim=1)
        value = self.value(vfe).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
