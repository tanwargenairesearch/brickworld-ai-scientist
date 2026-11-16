import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 32, "n_minibatches": 16, "gamma": 0.97, "gae_lambda": 0.95, "lr": 0.001, "ent_coef": 0.005}

W = 24
HID = 16
GL = 64
NBLOCKS = 2


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.film = nn.Linear(2 * c, c)

    def forward(self, h, B, Z):
        BZ, c, Y, X = h.shape
        h5 = h.view(B, Z, c, Y, X)
        g = torch.cat([h5.mean((1, 3, 4)), h5.amax((1, 3, 4))], dim=1)
        f = self.film(g)
        f = f[:, None, :].expand(B, Z, c).reshape(B * Z, c, 1, 1)
        return F.relu(h + self.conv(h) + f)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.nrest = int(spec.in_channels) - 4 * self.Z
        nin = 7 + self.nrest + 1
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.inp = nn.Conv2d(nin, W, 1)
        self.blocks = nn.ModuleList([Block(W) for _ in range(NBLOCKS)])
        self.hconvs = nn.ModuleList([nn.Conv2d(W, HID, (dy, dx)) for dy, dx in self.fps])
        self.souts = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.gouts = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        for o in self.souts:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, GL), nn.ReLU())
        self.done_head = nn.Linear(GL, 1)
        self.value_head = nn.Linear(GL, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        bb = F.pad(b, (0, 0, 0, 0, 1, 0), value=1.0)[:, 0:Z]
        ba = F.pad(b, (0, 0, 0, 0, 0, 1))[:, 1:Z + 1]
        ma = F.pad(m, (0, 0, 0, 0, 0, 1))[:, 1:Z + 1]
        per = torch.stack([b, t, m, e, bb, ba, ma], dim=2)
        rest = obs[:, 4 * Z:].unsqueeze(1).expand(B, Z, self.nrest, Y, X)
        zpos = (torch.arange(Z, device=obs.device).to(obs.dtype) / max(Z, 1)).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([per, rest, zpos], dim=2).reshape(B * Z, 7 + self.nrest + 1, Y, X)
        h = F.relu(self.inp(x))
        for blk in self.blocks:
            h = blk(h, B, Z)
        outs = []
        for (dy, dx), conv, so, go in zip(self.fps, self.hconvs, self.souts, self.gouts):
            q = F.pad(h, (0, dx - 1, 0, dy - 1))
            u = F.relu(conv(q))
            s = so(u).view(B, Z, Y, X)
            gt = go(u).view(B, Z, Y, X)
            w = torch.softmax(gt, dim=1)
            outs.append((w * s).sum(1))
        place = torch.stack(outs, dim=1).flatten(1)
        h5 = h.view(B, Z, W, Y, X)
        g = self.glob(torch.cat([h5.mean((1, 3, 4)), h5.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
