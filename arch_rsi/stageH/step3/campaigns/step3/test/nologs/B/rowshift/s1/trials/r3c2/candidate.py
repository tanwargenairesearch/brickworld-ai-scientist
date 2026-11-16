import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.995, "gae_lambda": 0.98, "ent_coef": 0.001, "lr": 0.001, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16}

H1 = 48
H = 32
FH = 24
G = 64


class CellNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = int(spec.in_channels) + 12 + self.Z
        self.t1 = nn.Conv2d(cin, H1, 1)
        self.t2 = nn.Conv2d(H1, H, 1)
        self.norm = CellNorm(H)
        self.convs = nn.ModuleList([nn.Conv2d(H, FH, (dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(FH, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.orthogonal_(o.weight, 0.01)
            nn.init.zeros_(o.bias)
        self.done_mlp = nn.Sequential(nn.Linear(2 * H, G), nn.ReLU(), nn.Linear(G, 1))
        self.value_mlp = nn.Sequential(nn.Linear(2 * H, G), nn.ReLU(), nn.Linear(G, 1))
        nn.init.orthogonal_(self.done_mlp[2].weight, 0.01)
        nn.init.zeros_(self.done_mlp[2].bias)
        nn.init.orthogonal_(self.value_mlp[2].weight, 1.0)
        nn.init.zeros_(self.value_mlp[2].bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Z, Y, X = self.Z, self.Y, self.X
        g4 = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        below = F.pad(g4, (0, 0, 0, 0, 1, 0))[:, :, :Z]
        above = F.pad(g4, (0, 0, 0, 0, 0, 1))[:, :, 1:]
        full = obs.unsqueeze(2).expand(B, obs.shape[1], Z, Y, X)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        tok = torch.cat([full, g4, below, above, eye], dim=1)
        tok = tok.permute(0, 2, 1, 3, 4).reshape(B * Z, -1, Y, X)
        h = F.relu(self.t1(tok))
        f = F.relu(self.norm(self.t2(h)))
        res = []
        for (dy, dx), conv, out in zip(self.fps, self.convs, self.outs):
            q = F.pad(f, (0, dx - 1, 0, dy - 1)) if dy * dx > 1 else f
            res.append(out(F.relu(conv(q))))
        s = torch.cat(res, dim=1).reshape(B, Z, len(self.fps), Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        fz = f.reshape(B, Z, H, Y, X)
        g = torch.cat([fz.mean((1, 3, 4)), fz.amax((1, 3, 4))], dim=1)
        logits = torch.cat([place, self.done_mlp(g)], dim=1)
        return logits, self.value_mlp(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
