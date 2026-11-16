import torch
import torch.nn as nn
import torch.nn.functional as F

W = 24
HID = 24
STEM = 32

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = 4 * 3 + 3 + 1
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.convs = nn.ModuleList([nn.Conv2d(W, HID, (dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.done_scale = nn.Parameter(torch.zeros(1))
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        per = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).transpose(1, 2)
        below = F.pad(per, (0, 0, 0, 0, 0, 0, 1, 0))[:, :Z]
        above = F.pad(per, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:]
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([per, below, above, gl, zc], dim=2).reshape(B * Z, 16, Y, X)
        h = self.stem(x)
        h = F.relu(h + self.c1(h))
        res = []
        for (dy, dx), conv, out in zip(self.fps, self.convs, self.outs):
            q = F.pad(h, (0, dx - 1, 0, dy - 1))
            s = out(F.relu(conv(q))).reshape(B, Z, Y * X)
            res.append(torch.logsumexp(s, dim=1))
        place = torch.cat(res, dim=1)
        hz = h.reshape(B, Z, W, Y, X)
        g = self.glob(torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1))
        lse = torch.logsumexp(place, dim=1, keepdim=True)
        done = self.done_head(g) - F.softplus(self.done_scale) * lse
        logits = torch.cat([place, done], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
