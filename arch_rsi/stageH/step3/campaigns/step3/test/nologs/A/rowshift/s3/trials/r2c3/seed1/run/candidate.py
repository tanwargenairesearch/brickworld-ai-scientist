import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.0007, "ent_coef": 0.003, "gamma": 0.95, "gae_lambda": 0.9}

H = 40
P = 16
M = 20
G = 64
NIN = 12


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.l1 = nn.Conv2d(NIN, H, 1)
        self.l2 = nn.Conv2d(H, H, 1)
        self.ctx = nn.Linear(2 * H, H)
        self.proj = nn.Conv2d(H, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, M, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(M, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.orthogonal_(o.weight, gain=0.01)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * H, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)
        nn.init.orthogonal_(self.done_head.weight, gain=0.01)
        nn.init.zeros_(self.done_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        one = obs.new_ones((B, 1, Y, X))
        zero = obs.new_zeros((B, 1, Y, X))
        b_below = torch.cat([one, b[:, :Z - 1]], dim=1)
        b_above = torch.cat([b[:, 1:], zero], dim=1)
        m_below = torch.cat([zero, m[:, :Z - 1]], dim=1)
        m_above = torch.cat([m[:, 1:], zero], dim=1)
        per = torch.stack([b, b_below, b_above, t, m, m_below, m_above, e], dim=2)
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype)
        zc = zc.view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([per, gl, zc], dim=2).reshape(B * Z, NIN, Y, X)

        h = F.relu(self.l1(x))
        hz = h.view(B, Z, H, Y, X)
        pooled = torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1)
        c = self.ctx(pooled)
        c = c[:, None, :].expand(B, Z, H).reshape(B * Z, H, 1, 1)
        h = h + F.relu(self.l2(h) + c)

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
            s = out(F.relu(mix(torch.cat([mx, mn, av], dim=1))))
            s = s.view(B, Z, Y, X)
            res.append(torch.logsumexp(s, dim=1))
        place = torch.stack(res, dim=1).flatten(1)

        hz = h.view(B, Z, H, Y, X)
        g = self.glob(torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
