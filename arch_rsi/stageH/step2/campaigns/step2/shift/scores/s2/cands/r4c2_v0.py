import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 256, "rollout_len": 16, "n_minibatches": 32, "ppo_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.02}

T = 32
HP = 64
G = 64
DONE_PRIOR = -2.0


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.nt = int(spec.n_types)
        self.np_ = int(spec.n_place)
        cin = spec.in_channels + self.nt
        self.t1 = nn.Conv2d(cin, T, 1)
        self.t2 = nn.Conv2d(T, T, 3, padding=1)
        self.patch = nn.ModuleList([nn.Conv2d(cin + T, HP, (dy, dx)) for dy, dx in self.fps])
        self.hid = nn.ModuleList([nn.Conv2d(HP, HP, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HP, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * T, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        m = mask[:, :self.np_].reshape(B, self.nt, Y, X).to(obs.dtype)
        x = torch.cat([obs, m], dim=1)
        t = F.relu(self.t1(x))
        t = F.relu(t + self.t2(t))
        f = torch.cat([x, t], dim=1)
        res = []
        for (dy, dx), pc, hd, out in zip(self.fps, self.patch, self.hid, self.outs):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            h = F.relu(pc(q))
            h = F.relu(h + hd(h))
            res.append(out(h))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([t.mean((2, 3)), t.amax((2, 3))], dim=1))
        done = self.done_head(g) + DONE_PRIOR
        logits = torch.cat([place, done], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
