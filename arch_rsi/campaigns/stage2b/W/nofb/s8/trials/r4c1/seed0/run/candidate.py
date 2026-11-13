import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "gae_lambda": 0.95}

W = 64
P = 32
H = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, W, 1), nn.ReLU(),
            nn.Conv2d(W, W, 1), nn.ReLU())
        self.ctx = nn.Conv2d(W, W, 3, padding=1)
        self.proj = nn.Conv2d(W, P, 1)
        self.bc = nn.Linear(W, P)
        heads = []
        for _ in self.footprints:
            out = nn.Conv2d(H, 1, 1)
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
            heads.append(nn.Sequential(nn.Conv2d(3 * P, H, 1), nn.ReLU(), out))
        self.heads = nn.ModuleList(heads)
        self.glob = nn.Sequential(nn.Linear(2 * W, W), nn.ReLU())
        self.done_head = nn.Linear(W, 1)
        self.value_head = nn.Linear(W, 1)

    def forward(self, obs, mask):
        f = self.stem(obs)
        f = f + F.relu(self.ctx(f))
        gl = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        g = F.relu(self.proj(f) + self.bc(gl)[:, :, None, None])
        outs = []
        for fp, head in zip(self.footprints, self.heads):
            dy, dx = fp
            gp = F.pad(g, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(gp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-gp, (dy, dx), stride=1)
            av = F.avg_pool2d(gp, (dy, dx), stride=1)
            outs.append(head(torch.cat([mx, mn, av], dim=1)).flatten(1))
        place = torch.cat(outs, dim=1)
        logits = torch.cat([place, self.done_head(gl)], dim=1)
        return logits, self.value_head(gl).squeeze(1)


def build_policy(spec):
    return Net(spec)
