import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 256, "rollout_len": 16, "n_minibatches": 32, "ppo_epochs": 1, "lr": 0.001, "ent_coef": 0.003, "gamma": 0.9, "gae_lambda": 0.9}

W = 48
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.P = int(spec.n_place)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = int(spec.in_channels) + self.T + 2
        self.s1 = nn.Conv2d(cin, 64, 1)
        self.s2 = nn.Conv2d(64, W, 1)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=2, dilation=2)
        self.c3 = nn.Conv2d(W, W, 3, padding=1)
        self.fconv = nn.ModuleList([nn.Conv2d(2 * W, HID, (dy, dx)) for dy, dx in self.fps])
        self.fout = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(4 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = self.Y
        X = self.X
        m = mask[:, :self.P].to(obs.dtype).reshape(B, self.T, Y, X)
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        inp = torch.cat([obs, m, yy, xx], dim=1)
        a = F.relu(self.s1(inp))
        s = F.relu(self.s2(a))
        h = F.relu(s + self.c1(s))
        h = F.relu(h + self.c2(h))
        h = F.relu(h + self.c3(h))
        f = torch.cat([s, h], dim=1)
        res = []
        for (dy, dx), cv, out in zip(self.fps, self.fconv, self.fout):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            res.append(out(F.relu(cv(q))))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
