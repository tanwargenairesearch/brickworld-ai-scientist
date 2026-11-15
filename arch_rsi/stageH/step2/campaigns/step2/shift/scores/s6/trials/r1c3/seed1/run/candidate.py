import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 64, "rollout_len": 16, "n_minibatches": 4, "ppo_epochs": 4, "gamma": 0.97}

H1 = 48
H2 = 32
PROJ = 16
HID = 24
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = 4 * (2 * Z - 1) + 3 + Z
        self.register_buffer("lay", torch.eye(Z).view(1, Z, Z, 1, 1))
        self.l1 = nn.Conv2d(cin, H1, 1)
        self.l2 = nn.Conv2d(H1, H2, 1)
        self.proj = nn.Conv2d(H2, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * H2 * Z, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        occ = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        occ = F.pad(occ, (0, 0, 0, 0, Z - 1, Z - 1))
        rel = torch.stack([occ[:, :, z:z + 2 * Z - 1] for z in range(Z)], dim=1)
        rel = rel.reshape(B, Z, 4 * (2 * Z - 1), Y, X)
        ext = obs[:, 4 * Z:].unsqueeze(1).expand(B, Z, C - 4 * Z, Y, X)
        lay = self.lay.expand(B, Z, Z, Y, X)
        inp = torch.cat([rel, ext, lay], dim=2).reshape(B * Z, -1, Y, X)
        h = F.relu(self.l1(inp))
        h = F.relu(self.l2(h))
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
            u = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(u))
        place = torch.cat(res, dim=1).reshape(B, Z, -1).logsumexp(1)
        g = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1).reshape(B, -1)
        g = self.glob(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
