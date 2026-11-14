import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0005, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.03, "gamma": 0.99, "gae_lambda": 0.95}

H1 = 48
P = 24
M = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = 4 * (2 * Z - 1) + 3 + Z
        self.c1 = nn.Conv2d(cin, H1, 1)
        self.c2 = nn.Conv2d(H1, H1, 1)
        self.proj = nn.Conv2d(H1, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, M, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(M, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * H1, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def slices(self, obs):
        B, C, Y, X = obs.shape
        Z = self.Z
        st = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        st = F.pad(st, (0, 0, 0, 0, Z - 1, Z - 1))
        rest = obs[:, 4 * Z:4 * Z + 3]
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype)
        out = []
        for z in range(Z):
            rel = st[:, :, z:z + 2 * Z - 1].reshape(B, 4 * (2 * Z - 1), Y, X)
            zc = eye[z].view(1, Z, 1, 1).expand(B, Z, Y, X)
            out.append(torch.cat([rel, rest, zc], dim=1))
        return torch.stack(out, dim=1).reshape(B * Z, -1, Y, X)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        Z = self.Z
        s = self.slices(obs)
        a = F.relu(self.c1(s))
        f = F.relu(a + self.c2(a))
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            sc = out(F.relu(mix(torch.cat([mx, mn, av], dim=1))))
            sc = sc.reshape(B, Z, Y * X)
            res.append(torch.logsumexp(sc, dim=1))
        place = torch.cat(res, dim=1)
        fz = f.reshape(B, Z, H1, Y, X)
        g = self.glob(torch.cat([fz.mean((1, 3, 4)), fz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
