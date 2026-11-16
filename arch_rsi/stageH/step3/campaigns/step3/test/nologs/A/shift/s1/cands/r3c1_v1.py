import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "lr": 0.001, "gamma": 0.95, "gae_lambda": 0.9, "ent_coef": 0.01, "n_minibatches": 16}

COLW = 64
W = 40
P = 24
HC = 24
H = 48
GH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.nt = int(spec.n_types)
        self.npl = int(spec.n_place)
        self.col = nn.Sequential(
            nn.Conv2d(spec.in_channels + self.nt, COLW, 1), nn.ReLU(),
            nn.Conv2d(COLW, W, 1), nn.ReLU())
        self.gin = nn.Linear(2 * W, W)
        self.b1 = nn.Conv2d(W, W, 3, padding=1)
        self.b2 = nn.Conv2d(W, W, 3, padding=1)
        self.hc = nn.Conv2d(W, HC, 3, padding=1)
        self.proj = nn.Conv2d(2 * W, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(P * dy * dx + HC, H, 1) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        m = mask[:, :self.npl].to(obs.dtype).reshape(B, self.nt, Y, X)
        c = self.col(torch.cat([obs, m], dim=1))
        g0 = self.gin(torch.cat([c.mean((2, 3)), c.amax((2, 3))], dim=1))
        f = F.relu(c + g0[:, :, None, None])
        f = F.relu(f + self.b1(f))
        f = F.relu(f + self.b2(f))
        u = F.relu(self.hc(f))
        p = F.relu(self.proj(torch.cat([c, f], dim=1)))
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                z = p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                z = torch.cat([q[:, :, oy:oy + Y, ox:ox + X] for oy in range(dy) for ox in range(dx)], dim=1)
            res.append(out(F.relu(mix(torch.cat([z, u], dim=1)))))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
