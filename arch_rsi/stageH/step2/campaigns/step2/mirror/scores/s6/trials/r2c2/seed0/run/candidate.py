import torch
import torch.nn as nn
import torch.nn.functional as F

HID = 64
K = 16
P = 16
H = 16

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.001, "gamma": 0.97, "ent_coef": 0.003}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.col = nn.Sequential(
            nn.Conv2d(spec.in_channels, HID, 1), nn.ReLU(),
            nn.Conv2d(HID, HID, 1), nn.ReLU())
        self.tolayer = nn.Conv2d(HID, self.Z * K, 1)
        self.sp1 = nn.Conv2d(K, K, 3, padding=1)
        self.mixz = nn.Conv2d(self.Z * K, self.Z * K, 1)
        self.sp2 = nn.Conv2d(K, K, 3, padding=1)
        self.proj = nn.Conv2d(K, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * HID + 2 * K, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        b, _, yy, xx = obs.shape
        z = self.Z
        c = self.col(obs)
        l = F.relu(self.tolayer(c))
        l = l.reshape(b * z, K, yy, xx)
        l = F.relu(l + self.sp1(l))
        l = l.reshape(b, z * K, yy, xx)
        l = F.relu(l + self.mixz(l))
        l = l.reshape(b * z, K, yy, xx)
        l = F.relu(l + self.sp2(l))
        p = self.proj(l)
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
        s = torch.cat(res, dim=1).reshape(b, z, len(self.fps), yy, xx)
        place = torch.logsumexp(s, dim=1).flatten(1)
        l5 = l.reshape(b, z, K, yy, xx)
        g_in = torch.cat([c.mean((2, 3)), c.amax((2, 3)), l5.mean((1, 3, 4)), l5.amax((1, 3, 4))], dim=1)
        g = self.glob(g_in)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
