import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
SKIP = 16
SQ = 32
GHID = 64
FRAMES = [0, 1]


def ortho_init(module, gain):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = nn.Conv2d(cin, SKIP, 1)
        self.out_ch = WIDTH + SKIP

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        k = F.relu(self.skip(obs))
        return torch.cat([x, k], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        self.K = max(max(a, b) for a, b in self.fps)
        self.trunk = Trunk(spec.in_channels)
        c = self.trunk.out_ch
        self.glob = nn.Sequential(nn.Linear(2 * c, GHID), nn.ReLU())
        self.done_head = nn.Linear(GHID, 1)
        self.value_head = nn.Linear(GHID, 1)
        self.type_bias = nn.Linear(GHID, T)
        self.dense = nn.Linear(GHID, spec.n_actions)
        nfr = len(FRAMES)
        self.sq = nn.ModuleList([nn.Conv2d(c, SQ, self.K) for _ in range(nfr)])
        self.outs = nn.ModuleList([nn.Conv2d(SQ, T, 1) for _ in range(nfr)])
        self.tmix = nn.Parameter(torch.zeros(nfr, T, T))
        self.sel = nn.Parameter(torch.zeros(nfr))
        ortho_init(self, math.sqrt(2))
        ortho_init(self.done_head, 0.01)
        ortho_init(self.type_bias, 0.01)
        ortho_init(self.dense, 0.01)
        ortho_init(self.outs, 0.01)
        ortho_init(self.value_head, 1.0)

    def forward(self, obs, mask):
        f = self.trunk(obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        tb = self.type_bias(g)[:, :, None, None]
        K = self.K
        outs = []
        for i, fr in enumerate(FRAMES):
            u = f.transpose(2, 3) if fr == 1 else f
            u = F.pad(u, (0, K - 1, 0, K - 1))
            l = self.outs[i](F.relu(self.sq[i](u)))
            l = l + torch.einsum('st,btyx->bsyx', self.tmix[i], l) + tb
            outs.append(l.flatten(1))
        w = F.softmax(5.0 * self.sel, dim=0)
        place = (torch.stack(outs, 1) * w[None, :, None]).sum(1)
        logits = torch.cat([place, self.done_head(g)], dim=1) + self.dense(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
