import torch
import torch.nn as nn
import torch.nn.functional as F

TR = 32
TOK = 32
PROJ = 16
HID = 16

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = int(spec.Z)
        self.Z = Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = spec.in_channels
        self.stem = nn.Sequential(
            nn.Conv2d(C, 48, 1), nn.ReLU(),
            nn.Conv2d(48, TR, 1), nn.ReLU())
        self.c1 = nn.Conv2d(TR, TR, 3, padding=1)
        self.c2 = nn.Conv2d(TR, TR, 3, padding=1)
        tin = 4 * (2 * Z - 1) + 3 + Z
        self.tok1 = nn.Conv2d(tin, TOK, 1)
        self.ctx = nn.Conv2d(TR, TOK, 1)
        self.tok2 = nn.Conv2d(TOK, TOK, 1)
        self.proj = nn.Conv2d(TOK, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * TR, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def tokens(self, obs):
        B, C, Y, X = obs.shape
        Z = self.Z
        groups = []
        for g in range(4):
            part = obs[:, g * Z:(g + 1) * Z]
            padded = F.pad(part, (0, 0, 0, 0, Z - 1, Z - 1))
            groups.append(torch.stack([padded[:, z:z + 2 * Z - 1] for z in range(Z)], dim=1))
        rel = torch.cat(groups, dim=2)
        common = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        t = torch.cat([rel, common, eye], dim=2)
        return t.reshape(B * Z, t.shape[2], Y, X)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        ctx = self.ctx(f)
        t = self.tok1(self.tokens(obs)).view(B, Z, TOK, Y, X) + ctx.unsqueeze(1)
        h = F.relu(t).reshape(B * Z, TOK, Y, X)
        h = F.relu(self.tok2(h))
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
            hh = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            hh = hh.view(B, Z, HID, Y, X).sum(1)
            res.append(out(hh))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
