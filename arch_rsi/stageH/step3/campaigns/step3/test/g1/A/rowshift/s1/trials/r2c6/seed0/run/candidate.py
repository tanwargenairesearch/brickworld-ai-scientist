import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CELL1 = 96
CELL2 = 64
SKIP = 16
DV = 16
NU = 4
PROJ = 32
HID = 48
GLOB = 64
DET = 16

RECIPE = {'ppo_epochs': 1, 'n_envs': 128, 'rollout_len': 64, 'n_minibatches': 32, 'lr': 0.0015, 'gamma': 0.95, 'gae_lambda': 0.9, 'ent_coef': 0.005}


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


class LayerFeat(nn.Module):
    def __init__(self, z, dv):
        super().__init__()
        self.z = z
        self.dv = dv
        self.l1 = nn.Linear(9, dv)
        self.l2 = nn.Linear(dv, dv)

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
        mbelow = F.pad(m, (0, 0, 0, 0, 1, 0), value=0.0)[:, :z]
        mabove = F.pad(m, (0, 0, 0, 0, 0, 1), value=0.0)[:, 1:]
        zpos = (torch.arange(z, device=obs.device, dtype=obs.dtype) / z).view(1, z, 1, 1).expand(bsz, z, ny, nx)
        u = torch.stack([b, t, m, e, below, above, mbelow, mabove, zpos], dim=-1)
        h = F.relu(self.l1(u))
        h = F.relu(self.l2(h))
        return h.permute(0, 1, 4, 2, 3).reshape(bsz, z * self.dv, ny, nx)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.z = spec.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        nt = len(self.fps)
        self.cell = nn.Sequential(
            nn.Conv2d(c, CELL1, 1), nn.ReLU(),
            nn.Conv2d(CELL1, CELL2, 1), nn.GroupNorm(8, CELL2), nn.ReLU())
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.scal = nn.Conv2d(CELL2, NU, 1)
        self.lay = LayerFeat(spec.Z, DV)
        self.proj = nn.Conv2d(CELL2 + SKIP, PROJ, 1)
        hin = 3 * DV + 3 * PROJ + 6 * NU
        self.mix = nn.Conv2d(hin, HID, 1)
        self.temb = nn.Parameter(torch.zeros(nt, HID))
        self.out = nn.Conv2d(HID, 1, 1)
        self.det = nn.Conv2d(c, DET, 1)
        self.glob = nn.Sequential(nn.Linear(2 * CELL2 + DET, GLOB), nn.ReLU())
        self.gproj = nn.Linear(GLOB, HID)
        self.type_bias = nn.Linear(GLOB, nt)
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        self.lbeta = nn.Parameter(torch.tensor(math.log(2.0 * spec.Z * spec.Z)))
        init_layers(self)
        nn.init.normal_(self.temb, std=0.1)
        init_out(self.gproj, 0.1)
        init_out(self.out, 0.01)
        init_out(self.type_bias, 0.01)
        init_out(self.done_head, 0.01)
        init_out(self.value_head, 1.0)

    def forward(self, obs, mask):
        z = self.z
        bsz = obs.shape[0]
        ny = obs.shape[2]
        nx = obs.shape[3]
        f = self.cell(obs)
        p = self.proj(torch.cat([f, self.skip(obs)], dim=1))
        u = self.scal(f)
        v = self.lay(obs)
        zc = (torch.arange(z, device=obs.device, dtype=obs.dtype) / z).view(1, z, 1, 1)
        beta = torch.exp(self.lbeta)
        d = F.relu(self.det(obs)).amax(dim=(2, 3))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), d], dim=1))
        gb = self.gproj(g).view(bsz, -1, 1, 1)
        tb = self.type_bias(g)
        res = []
        i = 0
        for (dy, dx) in self.fps:
            pu = fp_pool(u, dy, dx)
            land = pu[:, 0:1]
            w = torch.softmax(-beta * (zc - land) ** 2, dim=1)
            pv = fp_pool(v, dy, dx).reshape(bsz, 3, z, DV, ny, nx)
            gv = (pv * w.view(bsz, 1, z, 1, ny, nx)).sum(2).reshape(bsz, 3 * DV, ny, nx)
            hin = torch.cat([gv, fp_pool(p, dy, dx), pu, land - pu], dim=1)
            h = F.relu(self.mix(hin) + self.temb[i].view(1, -1, 1, 1) + gb)
            res.append(self.out(h) + tb[:, i].view(-1, 1, 1, 1))
            i += 1
        place = torch.cat(res, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
