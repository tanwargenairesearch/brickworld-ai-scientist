import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 3, "gae_lambda": 0.95}

W = 64
P = 32
H = 32
Q = 8


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, W, 1), nn.ReLU(),
            nn.Conv2d(W, W, 1), nn.ReLU())
        self.ctx = nn.Conv2d(W, W, 3, padding=1)
        self.proj = nn.Conv2d(W, P, 1)
        self.land = nn.Conv2d(W, self.Z, 1)
        self.lay = nn.Conv2d(W, self.Z * Q, 1)
        heads = []
        for _ in self.footprints:
            out = nn.Conv2d(H, 1, 1)
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
            heads.append(nn.Sequential(nn.Conv2d(3 * P + 2 * Q, H, 1), nn.ReLU(), out))
        self.heads = nn.ModuleList(heads)
        self.glob = nn.Sequential(nn.Linear(2 * W, W), nn.ReLU())
        self.done_head = nn.Linear(W, 1)
        self.value_head = nn.Linear(W, 1)

    def forward(self, obs, mask):
        f = self.stem(obs)
        f = f + F.relu(self.ctx(f))
        B, _, Y, X = f.shape
        Z = self.Z
        g = F.relu(self.proj(f))
        ld = self.land(f)
        ly = torch.tanh(self.lay(f))
        outs = []
        for fp, head in zip(self.footprints, self.heads):
            dy, dx = fp
            pad = (0, dx - 1, 0, dy - 1)
            gp = F.pad(g, pad)
            mx = F.max_pool2d(gp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-gp, (dy, dx), stride=1)
            av = F.avg_pool2d(gp, (dy, dx), stride=1)
            lp = F.max_pool2d(F.pad(ld, pad), (dy, dx), stride=1)
            a = torch.softmax(lp, dim=1).unsqueeze(2)
            yp = F.pad(ly, pad)
            ymn = (-F.max_pool2d(-yp, (dy, dx), stride=1)).reshape(B, Z, Q, Y, X)
            yav = F.avg_pool2d(yp, (dy, dx), stride=1).reshape(B, Z, Q, Y, X)
            sel = torch.cat([(a * ymn).sum(1), (a * yav).sum(1)], dim=1)
            outs.append(head(torch.cat([mx, mn, av, sel], dim=1)).flatten(1))
        place = torch.cat(outs, dim=1)
        gl = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(gl)], dim=1)
        return logits, self.value_head(gl).squeeze(1)


def build_policy(spec):
    return Net(spec)
