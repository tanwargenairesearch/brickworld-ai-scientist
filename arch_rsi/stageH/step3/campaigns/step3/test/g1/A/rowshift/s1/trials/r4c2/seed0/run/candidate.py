import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CELL1 = 128
CELL2 = 64
CONV = 32
SKIP = 16
PROJ = 24
HID = 32
GLOB = 64
DET = 16

RECIPE = {'ppo_epochs': 1, 'n_envs': 512, 'rollout_len': 16, 'n_minibatches': 64, 'lr': 0.003, 'gamma': 0.9, 'gae_lambda': 0.8, 'ent_coef': 0.0, 'clip_eps': 0.3}


def init_layers(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, math.sqrt(2.0))
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def init_out(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)


def win_feats(p, dy, dx):
    ny = p.shape[2]
    nx = p.shape[3]
    if dy * dx == 1:
        return torch.cat([p, p, p], dim=1)
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    parts = []
    for i in range(dy):
        for j in range(dx):
            parts.append(q[:, :, i:i + ny, j:j + nx])
    st = torch.stack(parts, 0)
    return torch.cat(parts + [st.amax(0), st.amin(0)], dim=1)


class WinHead(nn.Module):
    def __init__(self, fps, proj, hid):
        super().__init__()
        self.fps = fps
        self.mix1 = nn.ModuleList([nn.Conv2d((dy * dx + 2) * proj, hid, 1) for dy, dx in fps])
        self.mix2 = nn.ModuleList([nn.Conv2d(hid, hid, 1) for _ in fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in fps])

    def forward(self, p, gb):
        res = []
        for (dy, dx), m1, m2, o in zip(self.fps, self.mix1, self.mix2, self.outs):
            h = F.relu(m1(win_feats(p, dy, dx)) + gb)
            h = F.relu(m2(h))
            res.append(o(h))
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.cell = nn.Sequential(
            nn.Conv2d(c, CELL1, 1), nn.ReLU(),
            nn.Conv2d(CELL1, CELL2, 1), nn.GroupNorm(8, CELL2), nn.ReLU())
        self.r1 = nn.Conv2d(CELL2, CELL2, 1)
        self.nr1 = nn.GroupNorm(8, CELL2)
        self.r2 = nn.Conv2d(CELL2, CELL2, 1)
        self.nr2 = nn.GroupNorm(8, CELL2)
        self.conv = nn.Conv2d(CELL2, CONV, 3, padding=1)
        self.nc = nn.GroupNorm(8, CONV)
        self.skip = nn.Conv2d(c, SKIP, 1)
        fch = CELL2 + CONV
        self.proj = nn.Conv2d(fch + SKIP, PROJ, 1)
        self.head = WinHead(fps, PROJ, HID)
        self.det = nn.Conv2d(c, DET, 1)
        self.glob = nn.Sequential(nn.Linear(2 * fch + 2 * DET, GLOB), nn.ReLU())
        self.gproj = nn.Linear(GLOB, HID)
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        init_layers(self)
        for o in self.head.outs:
            init_out(o, 0.01)
        init_out(self.gproj, 0.1)
        init_out(self.done_head, 0.01)
        init_out(self.value_head, 1.0)

    def forward(self, obs, mask):
        bsz = obs.shape[0]
        a = self.cell(obs)
        a = F.relu(a + self.nr1(self.r1(a)))
        a = F.relu(a + self.nr2(self.r2(a)))
        b = F.relu(self.nc(self.conv(a)))
        f = torch.cat([a, b], dim=1)
        p = self.proj(torch.cat([f, self.skip(obs)], dim=1))
        dm = F.relu(self.det(obs))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), dm.amax((2, 3)), dm.mean((2, 3))], dim=1))
        gb = self.gproj(g).view(bsz, -1, 1, 1)
        place = self.head(p, gb).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
