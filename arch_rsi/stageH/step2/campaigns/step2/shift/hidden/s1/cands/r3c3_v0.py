import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
COL = 32
WIDTH = 32
H = 32
GLOB = 64
LIN = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.lin = nn.Conv2d(LIN, H, 1)
        self.tproj = nn.Conv2d(WIDTH + COL, H, 1)
        self.gin = nn.Linear(GLOB, H)
        self.win = nn.ModuleList([nn.Conv2d(H, H, (dy, dx)) for dy, dx in self.fps])
        self.mix = nn.ModuleList([nn.Conv2d(H, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 2, 1) for _ in self.fps])
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Sequential(nn.Linear(GLOB, GLOB), nn.ReLU(), nn.Linear(GLOB, 1))

    def layer_feats(self, obs):
        B, _, Y, X = obs.shape
        Z = self.Z
        L = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        below = F.pad(L, (0, 0, 0, 0, 1, 0))[:, :, :Z]
        above = F.pad(L, (0, 0, 0, 0, 0, 1))[:, :, 1:]
        cols = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1)
        zc = zc.view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        u = torch.cat([L, below, above, cols, zc], dim=1)
        return u.permute(0, 2, 1, 3, 4).reshape(B * Z, LIN, Y, X)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        ctx = self.tproj(torch.cat([s, f], dim=1)) + self.gin(g)[:, :, None, None]
        u = self.lin(self.layer_feats(obs)).view(B, Z, H, Y, X)
        h = F.relu(u + ctx.unsqueeze(1)).reshape(B * Z, H, Y, X)
        res = []
        for (dy, dx), win, mix, out in zip(self.fps, self.win, self.mix, self.outs):
            q = F.pad(h, (0, dx - 1, 0, dy - 1))
            o = out(F.relu(mix(F.relu(win(q))))).view(B, Z, 2, Y, X)
            sc = o[:, :, 0]
            w = torch.softmax(o[:, :, 1], dim=1)
            res.append((w * sc).sum(1).flatten(1))
        place = torch.cat(res, dim=1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
