"""r1c3 (height-shared layer encoder + footprint head) with ppo_epochs=2 for ~1.85x more samples."""
import torch
import torch.nn as nn
import torch.nn.functional as F

LW = 20
MIX = 48
WIDTH = 64
HEAD = 32
GLOB = 64

RECIPE = {"ppo_epochs": 2}


class FootprintHead(nn.Module):
    def __init__(self, c_in, footprints):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in footprints]
        self.convs = nn.ModuleList([nn.Conv2d(c_in, 1, (dy, dx)) for dy, dx in self.fps])
        self.cell = nn.Conv2d(c_in, len(self.fps), 1)

    def forward(self, h):
        cell = self.cell(h)
        outs = []
        for t, (dy, dx) in enumerate(self.fps):
            hp = F.pad(h, (0, dx - 1, 0, dy - 1))
            a = self.convs[t](hp)
            c = F.pad(cell[:, t:t + 1], (0, dx - 1, 0, dy - 1))
            m = -F.max_pool2d(-c, (dy, dx), stride=1)
            outs.append(a + m)
        return torch.cat(outs, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        c_layer = 3 * 4 + 3 + 1
        self.layer_body = nn.Sequential(
            nn.Conv2d(c_layer, LW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(LW, LW, 3, padding=1), nn.ReLU())
        self.zmix = nn.Conv2d(self.Z * LW, MIX, 1)
        self.fuse = nn.Conv2d(MIX + LW, WIDTH, 1)
        self.head_in = nn.Conv2d(WIDTH, HEAD, 1)
        self.place = FootprintHead(HEAD, spec.footprints)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        per = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).transpose(1, 2)
        pad = obs.new_zeros(B, 1, 4, Y, X)
        pp = torch.cat([pad, per, pad], dim=1)
        lay = torch.cat([pp[:, 0:Z], pp[:, 1:Z + 1], pp[:, 2:Z + 2]], dim=2)
        glob_in = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zpos = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([lay, glob_in, zpos], dim=2).reshape(B * Z, -1, Y, X)
        f = self.layer_body(x).reshape(B, Z, LW, Y, X)
        zm = F.relu(self.zmix(f.reshape(B, Z * LW, Y, X)))
        fmax = f.amax(dim=1)
        g2 = F.relu(self.fuse(torch.cat([zm, fmax], dim=1)))
        h = F.relu(self.head_in(g2))
        place = self.place(h).flatten(1)
        g = self.glob(torch.cat([g2.mean((2, 3)), g2.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
