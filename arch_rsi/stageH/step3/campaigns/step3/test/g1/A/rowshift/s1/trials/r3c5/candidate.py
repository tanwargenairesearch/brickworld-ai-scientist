import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
SKIP = 24
CTX = 16
PROJ = 32
CH = 24
K = 8
HID = 32
GLOB = 64
DET = 16

RECIPE = {'ppo_epochs': 1, 'n_envs': 128, 'rollout_len': 32, 'n_minibatches': 16, 'lr': 0.001, 'gamma': 0.97, 'gae_lambda': 0.9, 'ent_coef': 0.005}


def init_layers(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, math.sqrt(2.0))
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def init_out(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)


def coords(obs):
    bsz = obs.shape[0]
    ny = obs.shape[2]
    nx = obs.shape[3]
    yy = torch.linspace(-1.0, 1.0, ny, device=obs.device, dtype=obs.dtype).view(1, 1, ny, 1).expand(bsz, 1, ny, nx)
    xx = torch.linspace(-1.0, 1.0, nx, device=obs.device, dtype=obs.dtype).view(1, 1, 1, nx).expand(bsz, 1, ny, nx)
    return torch.cat([yy, xx], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(c + 2, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.GroupNorm(8, COL), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.ctxp = nn.Conv2d(WIDTH + SKIP, CTX, 1)
        cin = PROJ + SKIP + 2 * CTX
        self.chk1 = nn.ModuleList([nn.Conv2d(cin, CH, 1) for _ in self.fps])
        self.chk2 = nn.ModuleList([nn.Conv2d(CH, K, 1) for _ in self.fps])
        self.head1 = nn.ModuleList([nn.Conv2d(2 * K + 2 * CTX, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.det = nn.Conv2d(c, DET, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + 2 * DET, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        init_layers(self)
        for o in self.outs:
            init_out(o, 0.01)
        init_out(self.done_head, 0.01)
        init_out(self.value_head, 1.0)

    def forward(self, obs, mask):
        ny = obs.shape[2]
        nx = obs.shape[3]
        s = self.stem(torch.cat([obs, coords(obs)], dim=1))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        sk = self.skip(obs)
        q = torch.cat([self.proj(f), sk], dim=1)
        cx = self.ctxp(torch.cat([f, sk], dim=1))
        res = []
        for (dy, dx), k1, k2, hm, om in zip(self.fps, self.chk1, self.chk2, self.head1, self.outs):
            if dy * dx == 1:
                cmax = cx
                cmin = cx
                qp = q
            else:
                cp = F.pad(cx, (0, dx - 1, 0, dy - 1))
                cmax = F.max_pool2d(cp, (dy, dx), stride=1)
                cmin = -F.max_pool2d(-cp, (dy, dx), stride=1)
                qp = F.pad(q, (0, dx - 1, 0, dy - 1))
            ctx = torch.cat([cmax, cmin], dim=1)
            ss = []
            for i in range(dy):
                for j in range(dx):
                    qi = qp[:, :, i:i + ny, j:j + nx]
                    ss.append(k2(F.relu(k1(torch.cat([qi, ctx], dim=1)))))
            st = torch.stack(ss, 0)
            h = F.relu(hm(torch.cat([st.amin(0), st.mean(0), ctx], dim=1)))
            res.append(om(h))
        place = torch.cat(res, dim=1).flatten(1)
        dm = F.relu(self.det(obs))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), dm.amax((2, 3)), dm.mean((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
