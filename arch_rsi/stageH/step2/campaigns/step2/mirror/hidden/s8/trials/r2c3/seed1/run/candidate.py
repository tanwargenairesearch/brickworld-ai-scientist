import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "n_minibatches": 16, "ppo_epochs": 3}

STEM = 48
COLW = 32
D = 24
H = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU())
        self.c1 = nn.Conv2d(STEM, COLW, 3, padding=1)
        self.colp = nn.Conv2d(COLW, D, 1)
        self.lin = nn.Conv2d(16, D, 1)
        self.l1 = nn.Conv2d(D, D, 3, padding=1)
        self.fconv = nn.ModuleList([nn.Conv2d(D, H, (dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * COLW + 2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        col = F.relu(self.c1(self.stem(obs)))
        lay = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        lp = F.pad(lay, (0, 0, 0, 0, 1, 1))
        win = torch.cat([lp[:, :, 0:Z], lp[:, :, 1:Z + 1], lp[:, :, 2:Z + 2]], dim=1)
        glo = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1)).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        inp = torch.cat([win, glo, zc], dim=1)
        inp = inp.permute(0, 2, 1, 3, 4).reshape(B * Z, 16, Y, X)
        cp = self.colp(col)
        cp = cp.unsqueeze(1).expand(B, Z, D, Y, X).reshape(B * Z, D, Y, X)
        h = F.relu(self.lin(inp) + cp)
        h = F.relu(h + self.l1(h))
        res = []
        for (dy, dx), fc, out in zip(self.fps, self.fconv, self.outs):
            q = F.pad(h, (0, dx - 1, 0, dy - 1))
            o = out(F.relu(fc(q))).reshape(B, Z, 2, Y, X)
            w = torch.softmax(o[:, :, 1], dim=1)
            res.append((w * o[:, :, 0]).sum(1).flatten(1))
        place = torch.cat(res, dim=1)
        hz = h.reshape(B, Z, D, Y, X)
        g = self.glob(torch.cat([col.mean((2, 3)), col.amax((2, 3)),
                                 hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
