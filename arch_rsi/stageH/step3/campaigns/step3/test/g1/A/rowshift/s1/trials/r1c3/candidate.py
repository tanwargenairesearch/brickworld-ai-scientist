import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
BASE = 40
HEADS = 4
HDIM = 8
SKIP = 16
DET = 16
PROJ = 32
HID = 32
GLOB = 64

RECIPE = {'ppo_epochs': 2, 'n_envs': 128, 'rollout_len': 32, 'n_minibatches': 16, 'lr': 0.0015, 'gamma': 0.995, 'gae_lambda': 0.95, 'ent_coef': 0.005, 'clip_eps': 0.2}


def init_layers(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, math.sqrt(2.0))
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def init_out(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)


def fp_pool(p, dy, dx):
    if dy * dx == 1:
        return torch.cat([p, p, p], dim=1)
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return torch.cat([mx, mn, av], dim=1)


class LayerGather(nn.Module):
    def __init__(self, z, heads, dim):
        super().__init__()
        self.z = z
        self.heads = heads
        self.dim = dim
        self.key = nn.Linear(7, heads)
        self.val = nn.Linear(7, heads * dim)

    def forward(self, obs):
        z = self.z
        bsz = obs.shape[0]
        ny = obs.shape[2]
        nx = obs.shape[3]
        b = obs[:, 0:z]
        t = obs[:, z:2 * z]
        m = obs[:, 2 * z:3 * z]
        e = obs[:, 3 * z:4 * z]
        below = F.pad(b, (0, 0, 0, 0, 1, 0), value=1.0)[:, :z]
        above = F.pad(b, (0, 0, 0, 0, 0, 1), value=0.0)[:, 1:]
        zpos = (torch.arange(z, device=obs.device, dtype=obs.dtype) / z).view(1, z, 1, 1).expand(bsz, z, ny, nx)
        u = torch.stack([b, t, m, e, below, above, zpos], dim=-1)
        a = torch.softmax(self.key(u), dim=1)
        v = self.val(u).view(bsz, z, ny, nx, self.heads, self.dim)
        out = (a.unsqueeze(-1) * v).sum(1)
        return out.reshape(bsz, ny, nx, self.heads * self.dim).permute(0, 3, 1, 2).contiguous()


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        nt = len(self.fps)
        hd = HEADS * HDIM
        self.stem = nn.Sequential(
            nn.Conv2d(c, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, BASE, 1), nn.ReLU())
        self.gather = LayerGather(spec.Z, HEADS, HDIM)
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.fuse = nn.Conv2d(BASE + hd + SKIP, BASE, 1)
        self.nf = nn.GroupNorm(8, BASE)
        self.c1 = nn.Conv2d(BASE, BASE, 3, padding=1)
        self.n1 = nn.GroupNorm(8, BASE)
        self.c2 = nn.Conv2d(BASE, BASE, 3, padding=1)
        self.n2 = nn.GroupNorm(8, BASE)
        self.proj = nn.Conv2d(BASE + hd + SKIP, PROJ, 1)
        self.mix = nn.Conv2d(3 * PROJ, HID, 1)
        self.temb = nn.Parameter(torch.zeros(nt, HID))
        self.h2 = nn.Conv2d(HID, HID, 1)
        self.out = nn.Conv2d(HID, 1, 1)
        self.anchor = nn.Conv2d(PROJ, 1, 1)
        self.det = nn.Conv2d(c, DET, 1)
        self.glob = nn.Sequential(nn.Linear(2 * BASE + DET, GLOB), nn.ReLU())
        self.type_bias = nn.Linear(GLOB, nt)
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        init_layers(self)
        nn.init.normal_(self.temb, std=0.1)
        init_out(self.out, 0.01)
        init_out(self.anchor, 0.01)
        init_out(self.type_bias, 0.01)
        init_out(self.done_head, 0.01)
        init_out(self.value_head, 1.0)

    def forward(self, obs, mask):
        s = self.stem(obs)
        lg = self.gather(obs)
        sk = self.skip(obs)
        u = F.relu(self.nf(self.fuse(torch.cat([s, lg, sk], dim=1))))
        x = F.relu(self.n1(self.c1(u)))
        f = F.relu(u + self.n2(self.c2(x)))
        p = self.proj(torch.cat([f, lg, sk], dim=1))
        d = F.relu(self.det(obs)).amax(dim=(2, 3))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), d], dim=1))
        tb = self.type_bias(g)
        an = self.anchor(p)
        res = []
        i = 0
        for (dy, dx) in self.fps:
            h = F.relu(self.mix(fp_pool(p, dy, dx)) + self.temb[i].view(1, -1, 1, 1))
            h = F.relu(self.h2(h))
            res.append(self.out(h) + an + tb[:, i].view(-1, 1, 1, 1))
            i += 1
        place = torch.cat(res, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
