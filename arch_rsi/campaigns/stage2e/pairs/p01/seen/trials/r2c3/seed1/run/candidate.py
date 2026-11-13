import torch
import torch.nn as nn
import torch.nn.functional as F

W = 32
H = 16
RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "gamma": 0.97, "lr": 0.001}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.foots = [(int(f[0]), int(f[1])) for f in spec.footprints]
        self.stem = nn.Conv2d(13, W, 1)
        self.gstem = nn.Conv2d(3, W, 1)
        self.conv = nn.Conv2d(W, W, 3, padding=1)
        self.norm = nn.GroupNorm(4, W)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.bcast = nn.Linear(64, W)
        self.type_q = nn.ModuleList([nn.Conv2d(2 * W, H, 1) for _ in self.foots])
        self.mix = nn.Conv2d(3 * H, H, 1)
        self.out = nn.Conv2d(H, 2, 1)
        nn.init.orthogonal_(self.out.weight, 0.01)
        nn.init.zeros_(self.out.bias)
        self.done_head = nn.Linear(64, 1)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        per = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).transpose(1, 2)
        zero = torch.zeros_like(per[:, :1])
        below = torch.cat([zero, per[:, :-1]], dim=1)
        above = torch.cat([per[:, 1:], zero], dim=1)
        pos = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([per, below, above, pos], dim=2).reshape(B * Z, 13, Y, X)
        h0 = self.stem(x).view(B, Z, W, Y, X) + self.gstem(obs[:, 4 * Z:4 * Z + 3])[:, None]
        h0 = F.relu(h0).reshape(B * Z, W, Y, X)
        h = F.relu(h0 + self.norm(self.conv(h0)))
        hz = h.view(B, Z, W, Y, X)
        g = self.glob(torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1))
        h = F.relu(h + self.bcast(g).repeat_interleave(Z, dim=0)[:, :, None, None])
        outs = []
        for t, (dy, dx) in enumerate(self.foots):
            hp = F.pad(h, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(hp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-hp, (dy, dx), stride=1)
            q = F.relu(self.type_q[t](torch.cat([mx, mn], dim=1))).view(B, Z, H, Y, X)
            zq = torch.zeros_like(q[:, :1])
            cm = torch.cummax(q, dim=1).values
            bel = torch.cat([zq, cm[:, :-1]], dim=1)
            ca = torch.flip(torch.cummax(torch.flip(q, [1]), dim=1).values, [1])
            abv = torch.cat([ca[:, 1:], zq], dim=1)
            r = torch.cat([q, bel, abv], dim=2).reshape(B * Z, 3 * H, Y, X)
            o = self.out(F.relu(self.mix(r))).view(B, Z, 2, Y, X)
            w = torch.softmax(o[:, :, 1], dim=1)
            outs.append((w * o[:, :, 0]).sum(1))
        place = torch.stack(outs, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
