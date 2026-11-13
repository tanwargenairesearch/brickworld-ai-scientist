import math
import torch
import torch.nn as nn
import torch.nn.functional as F

AW = 48
CW = 24
VH = 64
P = 24
H = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.footprints = [(int(a), int(b)) for (a, b) in spec.footprints]
        self.n_types = spec.n_types
        self.actor = nn.Sequential(
            nn.Conv2d(c, AW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(AW, AW, 3, padding=1), nn.ReLU())
        self.reduce = nn.Conv2d(AW, P, 1)
        self.fp1 = nn.Conv2d(2 * P, H, 1)
        self.type_emb = nn.Parameter(torch.zeros(spec.n_types, H))
        self.fp_out = nn.Conv2d(H, 1, 1)
        self.glob = nn.Sequential(nn.Linear(2 * AW, AW), nn.ReLU())
        self.done_mlp = nn.Sequential(nn.Linear(AW + 2 * H, AW), nn.ReLU())
        self.done_head = nn.Linear(AW, 1)
        self.critic = nn.Sequential(
            nn.Conv2d(c, CW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(CW, CW, 3, padding=1), nn.ReLU())
        self.vmlp = nn.Sequential(nn.Linear(2 * CW, VH), nn.ReLU())
        self.value_head = nn.Linear(VH, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.fp_out.weight, gain=0.01)
        nn.init.orthogonal_(self.done_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        f = self.actor(obs)
        Yd, Xd = f.shape[2], f.shape[3]
        z = F.relu(self.reduce(f))
        pooled = []
        for (dy, dx) in self.footprints:
            zp = F.pad(z, (0, dx - 1, 0, dy - 1), value=0.0)
            mx = F.max_pool2d(zp, kernel_size=(dy, dx), stride=1)
            mn = F.avg_pool2d(zp, kernel_size=(dy, dx), stride=1)
            pooled.append(torch.cat([mx, mn], dim=1))
        q = torch.stack(pooled, dim=1)
        T = q.shape[1]
        q = q.reshape(B * T, 2 * P, Yd, Xd)
        h = self.fp1(q).reshape(B, T, H, Yd, Xd)
        h = F.relu(h + self.type_emb[None, :, :, None, None])
        place = self.fp_out(h.reshape(B * T, H, Yd, Xd)).reshape(B, T * Yd * Xd)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        hp = h.permute(0, 2, 1, 3, 4).reshape(B, H, T * Yd * Xd)
        dvec = torch.cat([g, hp.amax(2), hp.mean(2)], dim=1)
        done = self.done_head(self.done_mlp(dvec))
        logits = torch.cat([place, done], dim=1)
        hc = self.critic(obs)
        v = self.vmlp(torch.cat([hc.mean((2, 3)), hc.amax((2, 3))], dim=1))
        return logits, self.value_head(v).squeeze(1)


def build_policy(spec):
    return Net(spec)
