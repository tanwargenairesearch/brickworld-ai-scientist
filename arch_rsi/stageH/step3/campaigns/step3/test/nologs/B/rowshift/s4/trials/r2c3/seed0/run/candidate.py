import torch
import torch.nn as nn
import torch.nn.functional as F

D = 16
G = 8
HID = 16
GH = 64

RECIPE = {"ppo_epochs": 1, "gamma": 0.98, "gae_lambda": 0.95, "ent_coef": 0.01, "n_envs": 512, "rollout_len": 8, "n_minibatches": 32}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = int(spec.Z)
        self.Z = Z
        C = int(spec.in_channels)
        self.n_extra = C - 4 * Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        zc = torch.arange(Z, dtype=torch.float32) / max(Z - 1, 1)
        self.register_buffer("zc", zc.view(1, 1, Z, 1))
        self.flat = nn.Conv2d(C, G, 1)
        cin = 4 + self.n_extra + 1 + G
        self.e1 = nn.Conv2d(cin, D, (3, 1), padding=(1, 0))
        self.e2 = nn.Conv2d(3 * D, D, (3, 1), padding=(1, 0))
        self.s1 = nn.Conv2d(D, D, 3, padding=1)
        self.e3 = nn.Conv2d(D, D, (3, 1), padding=(1, 0))
        self.mix = nn.ModuleList([nn.Conv2d(3 * D, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.orthogonal_(o.weight, 0.01)
            nn.init.zeros_(o.bias)
        self.done = nn.Sequential(nn.Linear(2 * D, GH), nn.ReLU(), nn.Linear(GH, 1))
        self.value = nn.Sequential(nn.Linear(2 * D, GH), nn.ReLU(), nn.Linear(GH, 1))
        nn.init.orthogonal_(self.done[2].weight, 0.01)
        nn.init.zeros_(self.done[2].bias)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        N = Y * X
        lay = obs[:, :4 * Z].reshape(B, 4, Z, N)
        ext = obs[:, 4 * Z:].reshape(B, self.n_extra, 1, N).expand(B, self.n_extra, Z, N)
        fl = F.relu(self.flat(obs)).reshape(B, G, 1, N).expand(B, G, Z, N)
        zc = self.zc.expand(B, 1, Z, N)
        h = F.relu(self.e1(torch.cat([lay, ext, zc, fl], 1)))
        up = torch.flip(torch.cummax(torch.flip(h, [2]), 2)[0], [2])
        dn = torch.cummax(h, 2)[0]
        h = F.relu(self.e2(torch.cat([h, up, dn], 1)))
        s = h.reshape(B, D, Z, Y, X).permute(0, 2, 1, 3, 4).reshape(B * Z, D, Y, X)
        s = F.relu(s + self.s1(s))
        h = s.reshape(B, Z, D, N).permute(0, 2, 1, 3)
        h = F.relu(h + self.e3(h))
        f = h.permute(0, 2, 1, 3).reshape(B * Z, D, Y, X)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = f, f, f
            else:
                q = F.pad(f, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            o = out(F.relu(mix(torch.cat([mx, mn, av], 1))))
            o = o.reshape(B, Z, N)
            res.append(torch.logsumexp(o, 1))
        place = torch.cat(res, 1)
        g = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
        logits = torch.cat([place, self.done(g)], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
