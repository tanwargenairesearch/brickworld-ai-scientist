import torch
import torch.nn as nn
import torch.nn.functional as F

TRUNK = 48
CTX = 32
ENC = 48
LF = 32
P = 16
H = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = spec.in_channels
        Z = self.Z
        self.stem = nn.Sequential(nn.Conv2d(C, TRUNK, 1), nn.ReLU())
        self.c1 = nn.Conv2d(TRUNK, CTX, 3, padding=1)
        self.c2 = nn.Conv2d(CTX, CTX, 3, padding=1)
        nin = 4 * (2 * Z + 1) + 3 + Z
        self.enc1 = nn.Conv2d(nin, ENC, 1)
        self.enc2 = nn.Conv2d(ENC, LF, 1)
        self.ctx_proj = nn.Conv2d(CTX, LF, 1)
        self.proj = nn.Conv2d(LF, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * CTX, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def layer_inputs(self, obs):
        B, C, Y, X = obs.shape
        Z = self.Z
        groups = []
        for g in range(4):
            grp = obs[:, g * Z:(g + 1) * Z]
            padded = F.pad(grp, (0, 0, 0, 0, Z, Z))
            win = torch.stack([padded[:, z:z + 2 * Z + 1] for z in range(Z)], 1)
            groups.append(win)
        rest = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zid = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        inp = torch.cat(groups + [rest, zid], 2)
        return inp.reshape(B * Z, inp.shape[2], Y, X)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        h = F.relu(self.enc1(self.layer_inputs(obs)))
        h = self.enc2(h).view(B, Z, LF, Y, X) + self.ctx_proj(f).unsqueeze(1)
        h = F.relu(h).reshape(B * Z, LF, Y, X)
        p = self.proj(h)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            o = out(F.relu(mix(torch.cat([mx, mn, av], 1))))
            o = o.reshape(B, Z, Y * X)
            res.append(torch.logsumexp(o, dim=1))
        place = torch.cat(res, 1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
