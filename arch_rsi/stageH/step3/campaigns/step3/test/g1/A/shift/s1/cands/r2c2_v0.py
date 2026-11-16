import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.002, 'n_envs': 128, 'rollout_len': 64, 'n_minibatches': 16, 'ppo_epochs': 1, 'ent_coef': 0.005, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_eps': 0.25}

COLH = 96
COLW = 64
CTX = 32
SKIP = 32
PA = 32
PC = 32
HID = 32
H2 = 32
G = 64


def orth(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def coord_channels(obs):
    b, c, y, x = obs.shape
    ys = torch.linspace(-1.0, 1.0, y, device=obs.device, dtype=obs.dtype).view(1, 1, y, 1).expand(b, 1, y, x)
    xs = torch.linspace(-1.0, 1.0, x, device=obs.device, dtype=obs.dtype).view(1, 1, 1, x).expand(b, 1, y, x)
    return torch.cat([ys, xs], dim=1)


def fp_pool2(p, dy, dx):
    if dy * dx == 1:
        return p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    return F.max_pool2d(q, (dy, dx), stride=1), -F.max_pool2d(-q, (dy, dx), stride=1)


class RelHead(nn.Module):
    def __init__(self, cin, fps):
        super().__init__()
        self.fps = fps
        self.pa = nn.Conv2d(cin, PA, 1)
        self.pc = nn.Conv2d(cin, PC, 1)
        self.la = nn.ModuleList([nn.Conv2d(2 * PA, HID, 1) for _ in fps])
        self.lc = nn.ModuleList([nn.Conv2d(PC, HID, 1, bias=False) for _ in fps])
        self.l2 = nn.ModuleList([nn.Conv2d(HID, H2, 1) for _ in fps])
        self.outs = nn.ModuleList([nn.Conv2d(2 * H2, 1, 1) for _ in fps])

    def forward(self, f):
        ny, nx = f.shape[2], f.shape[3]
        a = self.pa(f)
        c = self.pc(f)
        res = []
        for k, (dy, dx) in enumerate(self.fps):
            amx, amn = fp_pool2(a, dy, dx)
            ha = self.la[k](torch.cat([amx, amn], dim=1))
            hc = F.pad(self.lc[k](c), (0, dx - 1, 0, dy - 1))
            zs = []
            for i in range(dy):
                for j in range(dx):
                    h = F.relu(ha + hc[:, :, i:i + ny, j:j + nx])
                    zs.append(F.relu(self.l2[k](h)))
            z = torch.stack(zs, dim=0)
            res.append(self.outs[k](torch.cat([z.amin(0), z.mean(0)], dim=1)))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.col = nn.Sequential(nn.Conv2d(cin + 2, COLH, 1), nn.ReLU(), nn.Conv2d(COLH, COLW, 1), nn.ReLU())
        self.ctx1 = nn.Conv2d(COLW, CTX, 3, padding=1)
        self.g1 = nn.GroupNorm(4, CTX)
        self.ctx2 = nn.Conv2d(CTX, CTX, 3, padding=1)
        self.g2 = nn.GroupNorm(4, CTX)
        self.skip = nn.Conv2d(cin, SKIP, 1)
        fc = COLW + CTX + SKIP
        self.head = RelHead(fc, fps)
        self.glob = nn.Sequential(nn.Linear(2 * fc, G), nn.ReLU(), nn.Linear(G, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Sequential(nn.Linear(G, G), nn.ReLU(), nn.Linear(G, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                orth(m, math.sqrt(2.0))
        for m in self.head.outs:
            orth(m, 0.01)
        orth(self.done_head, 0.01)
        orth(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        c = self.col(torch.cat([obs, coord_channels(obs)], dim=1))
        h = F.relu(self.g1(self.ctx1(c)))
        h = F.relu(self.g2(h + self.ctx2(h)))
        f = torch.cat([c, h, F.relu(self.skip(obs))], dim=1)
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
