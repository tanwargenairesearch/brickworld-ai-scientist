import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "lr": 0.001, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
W = 40
DC = 8
DH = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        self.n_place = int(spec.n_place)
        Y = int(spec.Y)
        X = int(spec.X)
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        k = 2 * max(max(a, b) for a, b in fps) - 1
        if k < 3:
            k = 3
        cin = int(spec.in_channels) + self.nt + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.c3 = nn.Conv2d(W, W, 3, padding=1)
        self.pout = nn.Conv2d(W, self.nt, k, padding=k // 2)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        self.dred = nn.Conv2d(W, DC, 1)
        self.dense = nn.Sequential(
            nn.Linear(DC * Y * X + 64, DH), nn.ReLU(),
            nn.Linear(DH, int(spec.n_actions)))

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        mp = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.nt, Y, X)
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        x = torch.cat([obs, mp, ys, xs], dim=1)
        s = self.stem(x)
        f = F.relu(s + self.c1(s))
        f = F.relu(f + self.c2(f))
        f = F.relu(f + self.c3(f))
        place = self.pout(f)[:, :, :Y, :X].flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        base = torch.cat([place, self.done_head(g)], dim=1)
        d = F.relu(self.dred(f)).flatten(1)
        logits = base + self.dense(torch.cat([d, g], dim=1))
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
