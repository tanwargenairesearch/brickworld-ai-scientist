import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 64, "rollout_len": 32, "n_minibatches": 16, "ppo_epochs": 1, "lr": 0.0007, "ent_coef": 0.003, "gamma": 0.97, "gae_lambda": 0.95}

DM = 64
HEADS = 4
FF = 128
PROJ = 24
HID = 24


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(DM)
        self.att = nn.MultiheadAttention(DM, HEADS, batch_first=True)
        self.n2 = nn.LayerNorm(DM)
        self.ff = nn.Sequential(nn.Linear(DM, FF), nn.ReLU(), nn.Linear(FF, DM))

    def forward(self, x):
        h = self.n1(x)
        a = self.att(h, h, h, need_weights=False)[0]
        x = x + a
        x = x + self.ff(self.n2(x))
        return x


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

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
        n_cells = int(spec.Y) * int(spec.X)
        self.embed = nn.Sequential(
            nn.Linear(int(spec.in_channels), DM), nn.ReLU(),
            nn.Linear(DM, DM))
        self.pos = nn.Parameter(torch.randn(1, n_cells, DM) * 0.02)
        self.blocks = nn.ModuleList([Block(), Block()])
        self.norm = nn.LayerNorm(DM)
        self.head = PoolHead(DM, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * DM, 64), nn.ReLU())
        self.done_head = nn.Linear(64 + 2, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        tok = obs.flatten(2).transpose(1, 2)
        x = self.embed(tok) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = F.relu(self.norm(x))
        f = x.transpose(1, 2).reshape(B, DM, Y, X)
        place = self.head(f)
        g = self.glob(torch.cat([x.mean(1), x.amax(1)], dim=1))
        pm = place.detach()
        dfeat = torch.cat([pm.amax(1, keepdim=True), pm.mean(1, keepdim=True)], dim=1)
        done = self.done_head(torch.cat([g, dfeat], dim=1))
        logits = torch.cat([place, done], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
