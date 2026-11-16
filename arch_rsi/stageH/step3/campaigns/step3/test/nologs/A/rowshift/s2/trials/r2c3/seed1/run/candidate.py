import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32, "lr": 0.001}

W = 24
P = 16
H = 16
GH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.n_extra = int(spec.in_channels) - 4 * self.Z
        self.in_per = 12 + self.n_extra + 1
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Conv2d(self.in_per, W, 1)
        self.conv = nn.Conv2d(W, W, 3, padding=1)
        self.ctx = nn.Linear(2 * W, W)
        self.proj = nn.Conv2d(W, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def layers(self, obs):
        B, C, Y, X = obs.shape
        Z = self.Z
        L = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 2, 1, 3, 4)
        zero = torch.zeros_like(L[:, :1])
        below = torch.cat([zero, L[:, :-1]], dim=1)
        above = torch.cat([L[:, 1:], zero], dim=1)
        g = obs[:, 4 * Z:].unsqueeze(1).expand(B, Z, self.n_extra, Y, X)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) + 0.5) / Z
        zc = zc.view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([L, below, above, g, zc], dim=2)
        return x.reshape(B * Z, self.in_per, Y, X)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        x = F.relu(self.stem(self.layers(obs)))
        h = F.relu(x + self.conv(x))
        hz = h.view(B, Z, W, Y, X)
        c = torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1)
        h = F.relu(h + self.ctx(c).repeat_interleave(Z, dim=0)[:, :, None, None])
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
            res.append(out(F.relu(mix(torch.cat([mx, mn, av], dim=1)))))
        s = torch.cat(res, dim=1).view(B, Z, len(self.fps), Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        hz = h.view(B, Z, W, Y, X)
        g = self.glob(torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
