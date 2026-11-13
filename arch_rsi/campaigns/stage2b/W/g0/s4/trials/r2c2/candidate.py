"""Height-shared layer encoder; placement scored per layer by a shared footprint head, then combined over z with a
learned softmax gate (soft landing-layer selection). Global features from a 1x1 z-mix."""
import torch
import torch.nn as nn
import torch.nn.functional as F

LW = 20
MIX = 48
HEAD = 20
GLOB = 64


class LayerFootprintHead(nn.Module):
    def __init__(self, c_in, footprints):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in footprints]
        self.convs = nn.ModuleList([nn.Conv2d(c_in, 2, (dy, dx)) for dy, dx in self.fps])
        self.cell = nn.Conv2d(c_in, len(self.fps), 1)

    def forward(self, h, B, Z):
        Y, X = h.shape[2], h.shape[3]
        cell = self.cell(h)
        outs = []
        for t, (dy, dx) in enumerate(self.fps):
            hp = F.pad(h, (0, dx - 1, 0, dy - 1))
            a = self.convs[t](hp)
            c = F.pad(cell[:, t:t + 1], (0, dx - 1, 0, dy - 1))
            m = -F.max_pool2d(-c, (dy, dx), stride=1)
            s = (a[:, 0:1] + m).reshape(B, Z, Y, X)
            w = torch.softmax(a[:, 1:2].reshape(B, Z, Y, X), dim=1)
            outs.append((w * s).sum(dim=1, keepdim=True))
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
        self.lay_in = nn.Conv2d(LW + MIX, HEAD, 1)
        self.place = LayerFootprintHead(HEAD, spec.footprints)
        self.glob = nn.Sequential(nn.Linear(2 * MIX, GLOB), nn.ReLU())
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
        f5 = self.layer_body(x).reshape(B, Z, LW, Y, X)
        zm = F.relu(self.zmix(f5.reshape(B, Z * LW, Y, X)))
        zmb = zm.unsqueeze(1).expand(B, Z, MIX, Y, X)
        h = F.relu(self.lay_in(torch.cat([f5, zmb], dim=2).reshape(B * Z, LW + MIX, Y, X)))
        place = self.place(h, B, Z).flatten(1)
        g = self.glob(torch.cat([zm.mean((2, 3)), zm.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
