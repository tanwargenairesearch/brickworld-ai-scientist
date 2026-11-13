import torch
import torch.nn as nn
import torch.nn.functional as F

WC = 24
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
        self.layer_body = nn.Sequential(
            nn.Conv2d(11, WC, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WC, WC, 3, padding=1), nn.ReLU())
        nin = WC + (Z + 6) + self.T
        self.head1 = nn.Conv2d(nin, HID, 1)
        self.head2 = nn.Conv2d(HID, 1, 1)
        nn.init.uniform_(self.head2.weight, -0.01, 0.01)
        nn.init.zeros_(self.head2.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WC + C, 64), nn.ReLU())
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
        return torch.cat([oh, tL, tA, gm, flat, tg], 1), oh

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        b = obs[:, :Z]
        tg = obs[:, Z:2 * Z]
        ms = obs[:, 2 * Z:3 * Z]
        ex = obs[:, 3 * Z:4 * Z]
        one = obs.new_ones(B, 1, Y, X)
        zero = obs.new_zeros(B, 1, Y, X)
        bb = torch.cat([one, b[:, :Z - 1]], 1)
        tb = torch.cat([one, tg[:, :Z - 1]], 1)
        ta = torch.cat([tg[:, 1:], zero], 1)
        hbz = obs[:, 4 * Z:4 * Z + 1].expand(B, Z, Y, X)
        htz = obs[:, 4 * Z + 1:4 * Z + 2].expand(B, Z, Y, X)
        tm = obs[:, 4 * Z + 2:4 * Z + 3].expand(B, Z, Y, X)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1).expand(B, Z, Y, X)
        per = torch.stack([b, tg, ms, ex, bb, tb, ta, hbz, htz, zc, tm], 2)
        Fl = self.layer_body(per.reshape(B * Z, 11, Y, X))
        feats = []
        for t, (dy, dx) in enumerate(self.fps):
            P = fp_pool(Fl, dy, dx, 'avg').reshape(B, Z, WC, Y, X)
            h, oh = self.hand(obs, dy, dx)
            sel = (oh[:, :Z].unsqueeze(2) * P).sum(1)
            to = obs.new_zeros(B, self.T, Y, X)
            to[:, t] = 1.0
            feats.append(torch.cat([sel, h, to], 1))
        x = torch.cat(feats, 0)
        p = self.head2(F.relu(self.head1(x)))
        place = p.reshape(self.T, B, Y * X).permute(1, 0, 2).reshape(B, -1)
        Fv = Fl.reshape(B, Z, WC, Y, X)
        g = self.glob(torch.cat([Fv.mean((1, 3, 4)), Fv.amax((1, 3, 4)), obs.mean((2, 3))], 1))
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
