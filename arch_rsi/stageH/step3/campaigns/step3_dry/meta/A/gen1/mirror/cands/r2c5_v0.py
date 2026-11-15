import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8}

W = 32
P = 16
H = 16
DS = 16
DONE_BIAS = 2.0


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.Y = spec.Y
        self.X = spec.X
        self.nt = spec.n_types
        self.n_place = spec.n_place
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = 12 + 3 + 1 + spec.n_types
        self.inp = nn.Conv2d(cin, W, 1)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.mixz = nn.Conv2d(2 * W, W, 1)
        self.proj = nn.Conv2d(W, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 2, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.zeros_(o.weight)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.dconv = nn.Conv2d(2 * spec.Z, DS, 1)
        self.done_head = nn.Linear(64 + 2 * DS, 1)
        self.done_bias = nn.Parameter(torch.full((1,), DONE_BIAS))
        self.value_head = nn.Linear(64 + 2 * DS, 1)
        nn.init.zeros_(self.done_head.weight)
        nn.init.zeros_(self.done_head.bias)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Z = self.Z
        Y = self.Y
        X = self.X
        L = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 2, 1, 3, 4)
        zero = torch.zeros_like(L[:, :1])
        below = torch.cat([zero, L[:, :-1]], dim=1)
        above = torch.cat([L[:, 1:], zero], dim=1)
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zpos = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1)).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, 1, self.nt, Y, X).expand(B, Z, self.nt, Y, X)
        x = torch.cat([L, below, above, gl, zpos, m], dim=2).reshape(B * Z, -1, Y, X)
        f = F.relu(self.inp(x))
        f = F.relu(f + self.c1(f))
        f5 = f.reshape(B, Z, W, Y, X)
        ctx = f5.amax(1, keepdim=True).expand(B, Z, W, Y, X)
        f = F.relu(f + self.mixz(torch.cat([f5, ctx], dim=2).reshape(B * Z, 2 * W, Y, X)))
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            o = out(h).reshape(B, Z, 2, Y, X)
            score = o[:, :, 0]
            gate = torch.softmax(o[:, :, 1], dim=1)
            res.append((score * gate).sum(1).flatten(1))
        place = torch.cat(res, dim=1)
        f5 = f.reshape(B, Z, W, Y, X)
        g = self.glob(torch.cat([f5.mean((1, 3, 4)), f5.amax((1, 3, 4))], dim=1))
        d = F.relu(self.dconv(obs[:, 2 * Z:4 * Z]))
        gd = torch.cat([g, d.amax((2, 3)), d.mean((2, 3))], dim=1)
        done = self.done_head(gd) + self.done_bias
        logits = torch.cat([place, done], dim=1)
        return logits, self.value_head(gd).squeeze(1)


def build_policy(spec):
    return Net(spec)
