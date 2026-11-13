import torch
import torch.nn as nn
import torch.nn.functional as F

W = 24
HID = 16
N_IN = 12


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for (a, b) in spec.footprints]
        self.conv1 = nn.Conv2d(N_IN, W, 3, padding=1)
        self.ctx = nn.Conv2d(W, W, 1)
        self.conv2 = nn.Conv2d(W, W, 3, padding=1)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * W, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in self.fps
        ])
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        Z = self.Z
        B, _, Y, X = obs.shape
        T = len(self.fps)
        build = obs[:, 0:Z]
        targ = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        b_below = F.pad(build, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        t_below = F.pad(targ, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        b_above = F.pad(build, (0, 0, 0, 0, 0, 1))[:, 1:]
        t_above = F.pad(targ, (0, 0, 0, 0, 0, 1))[:, 1:]
        hb = obs[:, 4 * Z:4 * Z + 1].expand(B, Z, Y, X)
        ht = obs[:, 4 * Z + 1:4 * Z + 2].expand(B, Z, Y, X)
        tm = obs[:, 4 * Z + 2:4 * Z + 3].expand(B, Z, Y, X)
        zi = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1).expand(B, Z, Y, X)
        x = torch.stack([build, targ, miss, extra, b_below, t_below, b_above, t_above, hb, ht, tm, zi], dim=2)
        x = x.reshape(B * Z, N_IN, Y, X)
        h = F.relu(self.conv1(x))
        hc = h.view(B, Z, W, Y, X).mean(1)
        c = self.ctx(hc).unsqueeze(1).expand(B, Z, W, Y, X).reshape(B * Z, W, Y, X)
        f = F.relu(self.conv2(h) + c)
        scores = []
        for (dy, dx), head in zip(self.fps, self.heads):
            p = F.pad(f, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(p, (dy, dx), stride=1)
            mn = -F.max_pool2d(-p, (dy, dx), stride=1)
            av = F.avg_pool2d(p, (dy, dx), stride=1)
            scores.append(head(torch.cat([av, mx, mn], dim=1)))
        s = torch.cat(scores, dim=1).view(B, Z, T, Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        f5 = f.view(B, Z, W, Y, X)
        g = torch.cat([f5.mean((1, 3, 4)), f5.amax((1, 3, 4))], dim=1)
        g = self.glob(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
