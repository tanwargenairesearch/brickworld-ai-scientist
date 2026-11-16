import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0005, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 3, "ent_coef": 0.04, "gamma": 0.99, "gae_lambda": 0.95}

K = 8
STEM = 64
W = 48
MH = 16
GH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem1 = nn.Conv2d(int(spec.in_channels) + 2, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, W, 1)
        self.ctx = nn.Conv2d(W, W, 3, padding=1)
        self.lay = nn.Conv2d(W, self.Z * K, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * K, MH, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(MH, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        yy = torch.linspace(0.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(0.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        s = F.relu(self.stem1(torch.cat([obs, yy, xx], dim=1)))
        s = F.relu(self.stem2(s))
        f = F.relu(s + self.ctx(s))
        p = self.lay(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            st = torch.stack([mx, mn, av], dim=1)
            st = st.reshape(B, 3, Z, K, Y, X).permute(0, 2, 1, 3, 4, 5).reshape(B * Z, 3 * K, Y, X)
            h = F.relu(mix(st))
            o = out(h).reshape(B, Z, Y, X)
            res.append(torch.logsumexp(o, dim=1))
        place = torch.stack(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
