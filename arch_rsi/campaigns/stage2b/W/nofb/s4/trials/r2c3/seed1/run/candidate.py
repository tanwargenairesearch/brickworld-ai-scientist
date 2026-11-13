import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 24
HID = 16
N_IN = 12
RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        W = WIDTH
        self.Z = spec.Z
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Conv2d(N_IN, W, 3, padding=1)
        self.col = nn.Conv2d(W, W, 1)
        self.glob_in = nn.Linear(2 * W, W)
        self.glob_out = nn.Linear(W, W)
        self.conv2 = nn.Conv2d(W, W, 3, padding=1)
        self.fp_proj = nn.Conv2d(W, HID, 1)
        self.fp_convs = nn.ModuleList([nn.Conv2d(W, HID, (dy, dx)) for dy, dx in self.footprints])
        self.fp_out = nn.ModuleList([nn.Conv2d(HID, 2, 1) for _ in self.footprints])
        for m in self.fp_out:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
        self.trunk = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        W = WIDTH
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        mi = obs[:, 2 * Z:3 * Z]
        ex = obs[:, 3 * Z:4 * Z]
        b_lo = F.pad(b, (0, 0, 0, 0, 1, 0))[:, :Z]
        t_lo = F.pad(t, (0, 0, 0, 0, 1, 0))[:, :Z]
        b_hi = F.pad(b, (0, 0, 0, 0, 0, 1))[:, 1:]
        t_hi = F.pad(t, (0, 0, 0, 0, 0, 1))[:, 1:]
        hb = obs[:, 4 * Z:4 * Z + 1].expand(B, Z, Y, X)
        ht = obs[:, 4 * Z + 1:4 * Z + 2].expand(B, Z, Y, X)
        tm = obs[:, 4 * Z + 2:4 * Z + 3].expand(B, Z, Y, X)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, Z, 1, 1).expand(B, Z, Y, X)
        x = torch.stack([b, t, mi, ex, b_lo, t_lo, b_hi, t_hi, hb, ht, tm, zc], dim=2)
        x = x.reshape(B * Z, N_IN, Y, X)
        h = F.relu(self.stem(x))
        hv = h.view(B, Z, W, Y, X)
        c = self.col(hv.amax(1))
        hv = hv + c[:, None]
        g = F.relu(self.glob_in(torch.cat([hv.mean((1, 3, 4)), hv.amax((1, 3, 4))], dim=1)))
        hv = hv + self.glob_out(g)[:, None, :, None, None]
        h = F.relu(hv).reshape(B * Z, W, Y, X)
        h = h + F.relu(self.conv2(h))
        p = self.fp_proj(h)
        outs = []
        for i, (dy, dx) in enumerate(self.footprints):
            hp = F.pad(h, (0, dx - 1, 0, dy - 1))
            a = self.fp_convs[i](hp)
            pp = F.pad(p, (0, dx - 1, 0, dy - 1))
            m = F.max_pool2d(pp, (dy, dx), stride=1)
            outs.append(self.fp_out[i](F.relu(a + m)))
        o = torch.stack(outs, dim=1).view(B, Z, len(self.footprints), 2, Y, X)
        s = o[:, :, :, 0]
        w = torch.softmax(o[:, :, :, 1], dim=1)
        place = (w * s).sum(1).flatten(1)
        hv2 = h.view(B, Z, W, Y, X)
        tr = self.trunk(torch.cat([hv2.mean((1, 3, 4)), hv2.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(tr)], dim=1)
        return logits, self.value_head(tr).squeeze(1)


def build_policy(spec):
    return Net(spec)
