import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 64, "rollout_len": 64, "n_minibatches": 64, "ppo_epochs": 2, "lr": 0.0005, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95}

D = 48
P = 16
H = 32


class FootHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.convs = nn.ModuleList([nn.Conv2d(c, hid, (dy, dx)) for dy, dx in self.fps])
        self.pmix = nn.ModuleList([nn.Conv2d(2 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), cv, pm, out in zip(self.fps, self.convs, self.pmix, self.outs):
            qf = F.pad(f, (0, dx - 1, 0, dy - 1))
            if dy * dx == 1:
                mx, mn = p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            h = F.relu(cv(qf) + pm(torch.cat([mx, mn], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_cells = int(spec.Y) * int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, 64, 1), nn.ReLU(),
            nn.Conv2d(64, D, 1), nn.ReLU())
        self.conv = nn.Conv2d(D, D, 3, padding=1)
        self.pos = nn.Parameter(torch.zeros(1, n_cells, D))
        nn.init.normal_(self.pos, std=0.02)
        self.attn = nn.TransformerEncoderLayer(d_model=D, nhead=4, dim_feedforward=2 * D, dropout=0.0, batch_first=True, norm_first=True)
        self.head = FootHead(D, spec.footprints, P, H)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        s = self.stem(obs)
        f = F.relu(s + self.conv(s))
        t = f.flatten(2).transpose(1, 2) + self.pos
        t = self.attn(t)
        f = F.relu(t.transpose(1, 2).reshape(B, D, Y, X))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
