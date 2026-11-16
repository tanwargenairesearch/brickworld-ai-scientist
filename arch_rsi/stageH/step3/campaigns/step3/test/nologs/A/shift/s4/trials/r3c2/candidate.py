import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "lr": 0.0005, "ent_coef": 0.003, "gamma": 0.97, "gae_lambda": 0.95, "clip_eps": 0.2}

W = 24
HH = 16
HID = 64
KIN = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.inp = nn.Conv2d(KIN, W, 1)
        self.colmix = nn.Conv2d(W, W, 1)
        self.gmix = nn.Linear(2 * W, W)
        self.conv = nn.Conv2d(W, W, 3, padding=1)
        self.fconv = nn.ModuleList([nn.Conv2d(W, HH, (dy, dx)) for dy, dx in self.fps])
        self.fout = nn.ModuleList([nn.Conv2d(HH, 1, 1) for _ in self.fps])
        for o in self.fout:
            nn.init.orthogonal_(o.weight, 0.01)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, HID), nn.ReLU())
        self.done_head = nn.Linear(HID, 1)
        self.value_head = nn.Linear(HID, 1)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)
        nn.init.orthogonal_(self.value_head.weight, 1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        lay = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], dim=2)
        below = F.pad(lay, (0, 0, 0, 0, 0, 0, 1, 0))[:, :Z]
        above = F.pad(lay, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:]
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1)).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        s = torch.cat([lay, below, above, gl, zc], dim=2).reshape(B * Z, KIN, Y, X)
        h = F.relu(self.inp(s))
        hv = h.view(B, Z, W, Y, X)
        col = self.colmix(hv.amax(1))
        gctx = self.gmix(torch.cat([hv.mean((1, 3, 4)), hv.amax((1, 3, 4))], dim=1))
        c = self.conv(h).view(B, Z, W, Y, X)
        xv = F.relu(hv + c + col.unsqueeze(1) + gctx[:, None, :, None, None])
        x = xv.reshape(B * Z, W, Y, X)
        res = []
        for (dy, dx), conv, out in zip(self.fps, self.fconv, self.fout):
            q = F.pad(x, (0, dx - 1, 0, dy - 1))
            res.append(out(F.relu(conv(q))))
        sc = torch.cat(res, dim=1).view(B, Z, len(self.fps), Y, X)
        place = torch.logsumexp(sc, dim=1).flatten(1)
        g = self.glob(torch.cat([xv.mean((1, 3, 4)), xv.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
