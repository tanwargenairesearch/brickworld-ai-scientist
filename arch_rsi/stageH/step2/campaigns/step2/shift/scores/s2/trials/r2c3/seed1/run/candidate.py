import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2}

T = 32
HP = 64
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = spec.in_channels
        self.t1 = nn.Conv2d(cin, T, 1)
        self.t2 = nn.Conv2d(T, T, 3, padding=1)
        self.patch = nn.ModuleList([nn.Conv2d(cin + T, HP, (dy, dx)) for dy, dx in self.fps])
        self.hid = nn.ModuleList([nn.Conv2d(HP, HP, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HP, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * T, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        t = F.relu(self.t1(obs))
        t = F.relu(t + self.t2(t))
        f = torch.cat([obs, t], dim=1)
        res = []
        for (dy, dx), pc, hd, out in zip(self.fps, self.patch, self.hid, self.outs):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            h = F.relu(pc(q))
            h = F.relu(h + hd(h))
            res.append(out(h))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([t.mean((2, 3)), t.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
