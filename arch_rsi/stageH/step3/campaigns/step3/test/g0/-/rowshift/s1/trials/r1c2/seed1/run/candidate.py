import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ent_coef": 0.005}

CTX = 32
LW = 32
PROJ = 16
HID = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = spec.in_channels
        self.ctx1 = nn.Conv2d(cin, CTX, 1)
        self.ctx2 = nn.Conv2d(CTX, CTX, 3, padding=1)
        self.ctx_to_layer = nn.Conv2d(CTX, LW, 1)
        self.lin = nn.Conv2d(12 + Z, LW, 1)
        self.l2 = nn.Conv2d(LW, LW, 1)
        self.proj = nn.Conv2d(LW, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * CTX, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        # 2D column context
        c = F.relu(self.ctx1(obs))
        f = F.relu(c + self.ctx2(c))
        ctx_l = self.ctx_to_layer(f)  # [B, LW, Y, X]
        # per-layer inputs
        grid = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        below = F.pad(grid, (0, 0, 0, 0, 1, 0))[:, :, :Z]
        above = F.pad(grid, (0, 0, 0, 0, 0, 1))[:, :, 1:]
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype)
        pos = eye.reshape(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        lay = torch.cat([grid, below, above, pos], dim=1)  # [B, 12+Z, Z, Y, X]
        lay = lay.permute(0, 2, 1, 3, 4).reshape(B * Z, 12 + Z, Y, X)
        ctx_b = ctx_l.unsqueeze(1).expand(B, Z, LW, Y, X).reshape(B * Z, LW, Y, X)
        h = F.relu(self.lin(lay) + ctx_b)
        h = F.relu(h + self.l2(h))
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
            o = out(u).reshape(B, Z, 2, Y, X)
            score = o[:, :, 0]
            gate = torch.softmax(o[:, :, 1], dim=1)
            res.append((gate * score).sum(1).flatten(1))
        place = torch.cat(res, dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
