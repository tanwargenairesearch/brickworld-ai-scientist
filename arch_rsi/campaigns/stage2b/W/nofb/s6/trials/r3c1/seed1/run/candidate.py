import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
LW = 32
P = 24
HEAD = 16
RECIPE = {"ppo_epochs": 1, "n_minibatches": 64, "gamma": 0.98, "gae_lambda": 0.95}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for (a, b) in spec.footprints]
        self.conv1 = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.bcast = nn.Linear(2 * WIDTH, WIDTH)
        self.layer_in = nn.Conv2d(11, LW, 1)
        self.trunk_to_layer = nn.Conv2d(WIDTH, LW, 1)
        self.mix = nn.Conv2d(2 * LW, P, 1)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * P, HEAD, 1), nn.ReLU(), nn.Conv2d(HEAD, 2, 1))
            for _ in self.fps
        ])
        self.base_head = nn.Conv2d(WIDTH, len(self.fps), 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        f = F.relu(self.conv1(obs))
        g0 = self.bcast(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        f = F.relu(self.conv2(f) + g0[:, :, None, None])

        build = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        bb = F.pad(build, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        tb = F.pad(tgt, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        mb = F.pad(miss, (0, 0, 0, 0, 1, 0), value=0.0)[:, :Z]
        eb = F.pad(extra, (0, 0, 0, 0, 1, 0), value=0.0)[:, :Z]
        zid = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1).expand(B, Z, Y, X)
        parts = [build, tgt, miss, extra, bb, tb, mb, eb, zid]
        per = torch.cat([t.unsqueeze(2) for t in parts], dim=2)
        hz = obs[:, 4 * Z:4 * Z + 2].unsqueeze(1).expand(B, Z, 2, Y, X)
        per = torch.cat([per, hz], dim=2).reshape(B * Z, 11, Y, X)

        e = self.layer_in(per).reshape(B, Z, LW, Y, X) + self.trunk_to_layer(f).unsqueeze(1)
        e = F.relu(e)
        below = torch.cummax(e, dim=1).values
        q = torch.cat([e, below], dim=2).reshape(B * Z, 2 * LW, Y, X)
        q = F.relu(self.mix(q))

        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            pp = F.pad(q, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(pp, kernel_size=(dy, dx), stride=1)
            mn = -F.max_pool2d(-pp, kernel_size=(dy, dx), stride=1)
            o = head(torch.cat([q, mx, mn], dim=1)).reshape(B, Z, 2, Y, X)
            w = torch.softmax(o[:, :, 0], dim=1)
            outs.append((w * o[:, :, 1]).sum(1))
        place = (torch.stack(outs, dim=1) + self.base_head(f)).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
