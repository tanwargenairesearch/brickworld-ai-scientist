import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}

D = 32
H = 16
GH = 64


def small_init(m):
    nn.init.orthogonal_(m.weight, gain=0.01)
    nn.init.zeros_(m.bias)


class CLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = spec.in_channels
        self.embed = nn.Conv2d(C, self.Z * D, 1)
        self.ctx = nn.Conv2d(C, D, 3, padding=1)
        self.norm1 = CLN(D)
        self.mix = nn.Conv2d(D, D, 1)
        self.norm2 = CLN(D)
        self.fconv = nn.ModuleList([nn.Conv2d(D, H, (dy, dx)) for dy, dx in self.fps])
        self.fout = nn.ModuleList([nn.Conv2d(H, 2, 1) for _ in self.fps])
        for m in self.fout:
            small_init(m)
        self.done_mlp = nn.Sequential(nn.Linear(2 * D, GH), nn.ReLU(), nn.Linear(GH, 1))
        self.val_mlp = nn.Sequential(nn.Linear(2 * D, GH), nn.ReLU(), nn.Linear(GH, 1))
        small_init(self.done_mlp[2])

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        e = self.embed(obs).view(B, Z, D, Y, X) + self.ctx(obs).unsqueeze(1)
        h = F.relu(e).reshape(B * Z, D, Y, X)
        h = h + self.mix(F.relu(self.norm1(h)))
        f = F.relu(self.norm2(h))
        res = []
        for (dy, dx), conv, out in zip(self.fps, self.fconv, self.fout):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            o = out(F.relu(conv(q))).view(B, Z, 2, Y, X)
            w = torch.softmax(o[:, :, 1], dim=1)
            res.append((w * o[:, :, 0]).sum(1))
        place = torch.stack(res, dim=1).flatten(1)
        fz = f.view(B, Z, D, Y, X)
        g = torch.cat([fz.mean((1, 3, 4)), fz.amax((1, 3, 4))], dim=1)
        logits = torch.cat([place, self.done_mlp(g)], dim=1)
        return logits, self.val_mlp(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
