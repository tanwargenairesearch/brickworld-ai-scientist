import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 256, "rollout_len": 16, "n_minibatches": 64, "lr": 0.001, "ent_coef": 0.005, "gamma": 0.97, "gae_lambda": 0.95}

TW = 32
TP = 12
LW = 16
CW = 4
FH = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Conv2d(spec.in_channels, TW, 1)
        self.c1 = nn.Conv2d(TW, TW, 3, padding=1)
        self.c2 = nn.Conv2d(TW, TW, 3, padding=1)
        self.tp = nn.Conv2d(TW, TP, 1)
        self.lin = nn.Conv2d(16 + TP, LW, 1)
        self.l2 = nn.Conv2d(LW, LW, 3, padding=1)
        self.cm = nn.Conv2d(LW, CW, 1)
        self.gate = nn.Conv2d(LW + CW, 1, 1)
        self.fconv = nn.ModuleList([nn.Conv2d(LW + CW, FH, (dy, dx)) for dy, dx in self.fps])
        self.fout = nn.ModuleList([nn.Conv2d(FH, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * TW + 2 * LW, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        t = F.relu(self.stem(obs))
        t = F.relu(t + self.c1(t))
        t = F.relu(t + self.c2(t))
        tp = self.tp(t)
        lay = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        pad = F.pad(lay, (0, 0, 0, 0, 1, 1))
        stk = torch.cat([pad[:, :, 0:Z], pad[:, :, 1:Z + 1], pad[:, :, 2:Z + 2]], dim=1)
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1)).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        tpe = tp.unsqueeze(2).expand(B, TP, Z, Y, X)
        x = torch.cat([stk, gl, zc, tpe], dim=1).permute(0, 2, 1, 3, 4).reshape(B * Z, 16 + TP, Y, X)
        h = F.relu(self.lin(x))
        h = F.relu(h + self.l2(h))
        c = self.cm(h).view(B, Z, CW, Y, X)
        cb = torch.cummax(c, dim=1).values.reshape(B * Z, CW, Y, X)
        hc = torch.cat([h, cb], dim=1)
        g = torch.sigmoid(self.gate(hc))
        res = []
        for (dy, dx), conv, out in zip(self.fps, self.fconv, self.fout):
            q = F.pad(hc, (0, dx - 1, 0, dy - 1))
            s = out(F.relu(conv(q))).view(B, Z, Y, X)
            gq = F.pad(g, (0, dx - 1, 0, dy - 1))
            m = F.max_pool2d(gq, (dy, dx), stride=1).view(B, Z, Y, X)
            mprev = torch.cat([torch.ones_like(m[:, :1]), m[:, :-1]], dim=1)
            land = F.relu(mprev - m)
            res.append((land * s).sum(1))
        place = torch.stack(res, dim=1).flatten(1)
        h5 = h.view(B, Z, LW, Y, X)
        gf = torch.cat([t.mean((2, 3)), t.amax((2, 3)), h5.mean(dim=(1, 3, 4)), h5.amax(dim=(1, 3, 4))], dim=1)
        gg = self.glob(gf)
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
