import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 256, "rollout_len": 16, "n_minibatches": 64, "lr": 0.001, "ent_coef": 0.005, "gamma": 0.97, "gae_lambda": 0.95}

WIDTH = 32
PROJ = 24
HID = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        W = WIDTH
        self.inp = nn.Conv2d(16, W, 1)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 1)
        self.col = nn.Conv2d(W, W, 1)
        self.proj = nn.Conv2d(W, PROJ, 1)
        self.convs = nn.ModuleList([nn.Conv2d(PROJ, HID, (dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        lay = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        pad = F.pad(lay, (0, 0, 0, 0, 1, 1))
        stk = torch.cat([pad[:, :, 0:Z], pad[:, :, 1:Z + 1], pad[:, :, 2:Z + 2]], dim=1)
        g = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1)).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        x = torch.cat([stk, g, zc], dim=1).permute(0, 2, 1, 3, 4).reshape(B * Z, 16, Y, X)
        h = F.relu(self.inp(x))
        h = F.relu(self.c1(h))
        W = h.shape[1]
        colf = self.col(h.view(B, Z, W, Y, X).amax(1))
        colb = colf.unsqueeze(1).expand(B, Z, W, Y, X).reshape(B * Z, W, Y, X)
        h = F.relu(h + self.c2(h) + colb)
        p = self.proj(h)
        res = []
        for (dy, dx), conv, out in zip(self.fps, self.convs, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            a = F.relu(conv(q))
            o = out(a).view(B, Z, Y, X)
            res.append(torch.logsumexp(o, dim=1))
        place = torch.stack(res, dim=1).flatten(1)
        h5 = h.view(B, Z, W, Y, X)
        gf = torch.cat([h5.mean(dim=(1, 3, 4)), h5.amax(dim=(1, 3, 4))], dim=1)
        gg = self.glob(gf)
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
