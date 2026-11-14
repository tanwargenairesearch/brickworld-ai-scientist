import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 64, "rollout_len": 16, "n_minibatches": 4, "ppo_epochs": 4, "ent_coef": 0.005, "gamma": 0.97, "gae_lambda": 0.95}

G = 32
H = 24
M = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        c = spec.in_channels
        self.g1 = nn.Conv2d(c, G, 1)
        self.g2 = nn.Conv2d(G, G, 3, padding=1)
        self.q_obs = nn.Conv2d(c, H, 1)
        self.q_ctx = nn.Conv2d(G, H, 1)
        self.q_z = nn.Linear(self.Z, H, bias=False)
        self.q2 = nn.Conv2d(H, H, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * H, M, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(M, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * G, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        Z = self.Z
        a = F.relu(self.g1(obs))
        f = F.relu(a + self.g2(a))
        base = self.q_obs(obs) + self.q_ctx(f)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype)
        ze = self.q_z(eye)
        h = F.relu(base.unsqueeze(1) + ze.view(1, Z, -1, 1, 1))
        h = h.reshape(B * Z, -1, Y, X)
        p = F.relu(self.q2(h))
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            s = out(F.relu(mix(torch.cat([mx, mn, av], dim=1))))
            s = s.reshape(B, Z, Y * X)
            res.append(torch.logsumexp(s, dim=1))
        place = torch.cat(res, dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
