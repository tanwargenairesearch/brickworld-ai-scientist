import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.01}

TR = 32
LW = 32
PROJ = 16
HID = 24
DONE_PRIOR = -3.0


def pool3(p, dy, dx):
    if dy * dx == 1:
        return [p, p, p]
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return [mx, mn, av]


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = spec.in_channels
        self.stem = nn.Conv2d(C, TR, 1)
        self.ctx = nn.Conv2d(TR, TR, 3, padding=1)
        self.ctx_proj = nn.Conv2d(TR, LW, 1)
        self.lin = nn.Conv2d(11, LW, 1)
        self.l2 = nn.Conv2d(LW, LW, 1)
        self.proj = nn.Conv2d(LW, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * TR + LW, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = F.relu(self.stem(obs))
        c = F.relu(s + self.ctx(s))
        b = obs[:, 0:Z]
        b_below = F.pad(b, (0, 0, 0, 0, 1, 0), value=1.0)[:, 0:Z]
        t = obs[:, Z:2 * Z]
        t_below = F.pad(t, (0, 0, 0, 0, 1, 0), value=1.0)[:, 0:Z]
        m = obs[:, 2 * Z:3 * Z]
        m_above = F.pad(m, (0, 0, 0, 0, 0, 1))[:, 1:Z + 1]
        e = obs[:, 3 * Z:4 * Z]
        per = torch.stack([b, b_below, t, t_below, m, m_above, e], dim=2)
        ex = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        pos = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        inp = torch.cat([per, ex, pos], dim=2).reshape(B * Z, 11, Y, X)
        cc = self.ctx_proj(c).unsqueeze(1).expand(B, Z, LW, Y, X).reshape(B * Z, LW, Y, X)
        h = F.relu(self.lin(inp) + cc)
        h = F.relu(self.l2(h))
        p = self.proj(h)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            hh = F.relu(mix(torch.cat(pool3(p, dy, dx), dim=1)))
            o = out(hh).reshape(B, Z, Y, X)
            res.append(torch.logsumexp(o, dim=1).flatten(1))
        place = torch.cat(res, dim=1)
        hm = h.reshape(B, Z, LW, Y, X).mean((1, 3, 4))
        g = self.glob(torch.cat([c.mean((2, 3)), c.amax((2, 3)), hm], dim=1))
        d = self.done_head(g) + DONE_PRIOR
        place_lp = F.log_softmax(place, dim=1) + F.logsigmoid(-d)
        done_lp = F.logsigmoid(d)
        logits = torch.cat([place_lp, done_lp], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
