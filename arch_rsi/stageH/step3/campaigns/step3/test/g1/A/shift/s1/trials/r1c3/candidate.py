import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.0015, 'n_envs': 64, 'rollout_len': 32, 'n_minibatches': 8, 'ppo_epochs': 3, 'ent_coef': 0.01, 'gamma': 0.97, 'gae_lambda': 0.95, 'clip_eps': 0.25}

COLH = 96
COLW = 64
CTX = 32
P = 32
HID = 32
G = 64


def orth(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def coord_channels(obs):
    b, c, y, x = obs.shape
    ys = torch.linspace(-1.0, 1.0, y, device=obs.device, dtype=obs.dtype).view(1, 1, y, 1).expand(b, 1, y, x)
    xs = torch.linspace(-1.0, 1.0, x, device=obs.device, dtype=obs.dtype).view(1, 1, 1, x).expand(b, 1, y, x)
    return torch.cat([ys, xs], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.col = nn.Sequential(nn.Conv2d(cin + 2, COLH, 1), nn.ReLU(), nn.Conv2d(COLH, COLW, 1), nn.ReLU())
        self.ctx1 = nn.Conv2d(COLW, CTX, 3, padding=1)
        self.g1 = nn.GroupNorm(4, CTX)
        self.ctx2 = nn.Conv2d(CTX, CTX, 3, padding=1)
        self.g2 = nn.GroupNorm(4, CTX)
        self.proj = nn.Conv2d(COLW + CTX + cin, P, 1)
        self.fconv = nn.ModuleList([nn.Conv2d(P, HID, (dy, dx)) for dy, dx in self.fps])
        self.pmix = nn.ModuleList([nn.Conv2d(2 * P, HID, 1) for _ in self.fps])
        self.mid = nn.ModuleList([nn.Conv2d(HID, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * COLW + 2 * CTX, G), nn.ReLU(), nn.Linear(G, G), nn.ReLU())
        self.dense = nn.Linear(G, spec.n_place)
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Sequential(nn.Linear(G, G), nn.ReLU(), nn.Linear(G, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                orth(m, math.sqrt(2.0))
        for m in self.outs:
            orth(m, 0.01)
        orth(self.dense, 0.01)
        orth(self.done_head, 0.01)
        orth(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        c = self.col(torch.cat([obs, coord_channels(obs)], dim=1))
        h = F.relu(self.g1(self.ctx1(c)))
        h = F.relu(self.g2(h + self.ctx2(h)))
        p = self.proj(torch.cat([c, h, obs], dim=1))
        g = self.glob(torch.cat([c.mean((2, 3)), c.amax((2, 3)), h.mean((2, 3)), h.amax((2, 3))], dim=1))
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            a = self.fconv[i](q)
            if dy * dx == 1:
                mx, mn = p, p
            else:
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            z = F.relu(a + self.pmix[i](torch.cat([mx, mn], dim=1)))
            z = F.relu(self.mid[i](z))
            res.append(self.outs[i](z))
        place = torch.cat(res, dim=1).flatten(1) + self.dense(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
