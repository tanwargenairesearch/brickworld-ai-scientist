import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
LW = 32
PROJ = 32
HEAD = 32
RECIPE = {"ppo_epochs": 1, "n_minibatches": 64, "n_envs": 128, "rollout_len": 32, "gamma": 0.98, "gae_lambda": 0.95}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for (a, b) in spec.footprints]
        self.layer_conv1 = nn.Conv2d(12, LW, 3, padding=1)
        self.layer_conv2 = nn.Conv2d(LW, LW, 1)
        self.zmix = nn.Conv2d(self.Z * LW + LW, WIDTH, 1)
        self.conv1 = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.bcast = nn.Linear(2 * WIDTH, WIDTH)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * PROJ, HEAD, 1), nn.ReLU(), nn.Conv2d(HEAD, 1, 1))
            for _ in self.fps
        ])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        build = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        rest = obs[:, 4 * Z:4 * Z + 3]
        bb = F.pad(build, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        tb = F.pad(tgt, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        mb = F.pad(miss, (0, 0, 0, 0, 1, 0), value=0.0)[:, :Z]
        eb = F.pad(extra, (0, 0, 0, 0, 1, 0), value=0.0)[:, :Z]
        zid = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1).expand(B, Z, Y, X)
        parts = [build, tgt, miss, extra, bb, tb, mb, eb, zid]
        per = torch.cat([t.unsqueeze(2) for t in parts], dim=2)
        restz = rest.unsqueeze(1).expand(B, Z, 3, Y, X)
        per = torch.cat([per, restz], dim=2).reshape(B * Z, 12, Y, X)
        h = F.relu(self.layer_conv1(per))
        h = F.relu(self.layer_conv2(h))
        h = h.reshape(B, Z, LW, Y, X)
        hz = torch.cat([h.reshape(B, Z * LW, Y, X), h.amax(1)], dim=1)
        lz = self.zmix(hz)
        f = F.relu(self.conv1(obs) + lz)
        g0 = self.bcast(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        f = F.relu(self.conv2(f) + g0[:, :, None, None])
        p = F.relu(self.proj(f))
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            pp = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(pp, kernel_size=(dy, dx), stride=1)
            mn = -F.max_pool2d(-pp, kernel_size=(dy, dx), stride=1)
            outs.append(head(torch.cat([p, mx, mn], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
