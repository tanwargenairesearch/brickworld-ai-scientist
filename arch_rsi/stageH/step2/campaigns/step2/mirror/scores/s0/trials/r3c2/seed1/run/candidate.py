import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.0015, "ent_coef": 0.005}

H1 = 32
L = 24
TR = 24
PROJ = 16
HID = 16


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.orthogonal_(o.weight, 0.01)
            nn.init.zeros_(o.bias)

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        c = spec.in_channels
        nin = 12 + 3 + 1
        self.l1 = nn.Conv2d(nin, H1, 1)
        self.l2 = nn.Conv2d(H1, L, 1)
        self.l3 = nn.Conv2d(3 * L, L, 1)
        self.tin = nn.Conv2d(c, TR, 1)
        self.tc = nn.Conv2d(TR, TR, 3, padding=1)
        self.ctx = nn.Conv2d(TR, L, 1)
        self.head = PoolHead(L, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * TR + 2 * L, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        g4 = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        pz = F.pad(g4, (0, 0, 0, 0, 1, 1))
        per = torch.cat([pz[:, :, 0:Z], pz[:, :, 1:Z + 1], pz[:, :, 2:Z + 2]], dim=1)
        planes = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zpos = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        inp = torch.cat([per, planes, zpos], dim=1)
        inp = inp.permute(0, 2, 1, 3, 4).reshape(B * Z, 16, Y, X)
        h = F.relu(self.l1(inp))
        t = F.relu(self.l2(h))
        tv = t.view(B, Z, L, Y, X)
        zero = torch.zeros_like(tv[:, :1])
        below = torch.cummax(tv, dim=1).values
        below = torch.cat([zero, below[:, :-1]], dim=1)
        above = torch.flip(torch.cummax(torch.flip(tv, [1]), dim=1).values, [1])
        above = torch.cat([above[:, 1:], zero], dim=1)
        a = F.relu(self.tin(obs))
        f = F.relu(a + self.tc(a))
        cx = self.ctx(f).unsqueeze(1).expand(B, Z, L, Y, X).reshape(B * Z, L, Y, X)
        u = torch.cat([tv, below, above], dim=2).reshape(B * Z, 3 * L, Y, X)
        u = F.relu(self.l3(u) + cx)
        s = self.head(u).view(B, Z, -1)
        place = torch.logsumexp(s, dim=1)
        uv = u.view(B, Z, L, Y, X)
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3)), uv.mean((1, 3, 4)), uv.amax(dim=(1, 3, 4))], dim=1)
        g = self.glob(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
