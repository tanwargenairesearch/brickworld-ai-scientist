import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "gamma": 0.995, "gae_lambda": 0.95, "ent_coef": 0.01, "lr": 0.0008, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16}

STEM = 48
W = 40
HID = 48
G = 64
K = 6


class CellNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = CellNorm(c)
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv(F.relu(self.norm(x)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.ok = self.T * self.Y * self.X == self.n_place
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1))
        self.blocks = nn.Sequential(Block(W), Block(W))
        self.fnorm = CellNorm(W)
        self.hid = nn.Conv2d(W, HID, 3, padding=1)
        if self.ok:
            self.heads = nn.ModuleList([nn.Conv2d(HID, self.T, 1) for _ in range(K + 1)])
            for o in self.heads:
                nn.init.orthogonal_(o.weight, 0.01)
                nn.init.zeros_(o.bias)
        else:
            self.cell = nn.Conv2d(HID, 8, 1)
            self.fallback = nn.Linear(8 * self.Y * self.X, self.n_place)
            nn.init.orthogonal_(self.fallback.weight, 0.01)
            nn.init.zeros_(self.fallback.bias)
        self.done_mlp = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU(), nn.Linear(G, 1))
        self.done2 = nn.Linear(2 * W, 1)
        self.value_mlp = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU(), nn.Linear(G, 1))
        nn.init.orthogonal_(self.done_mlp[2].weight, 0.01)
        nn.init.zeros_(self.done_mlp[2].bias)
        nn.init.orthogonal_(self.done2.weight, 0.01)
        nn.init.zeros_(self.done2.bias)
        nn.init.orthogonal_(self.value_mlp[2].weight, 1.0)
        nn.init.zeros_(self.value_mlp[2].bias)

    def forward(self, obs, mask):
        x = self.stem(obs)
        x = self.blocks(x)
        f = F.relu(self.fnorm(x))
        h = F.relu(self.hid(f))
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        d = self.done_mlp(g)
        if self.ok:
            s = [hd(h) for hd in self.heads]
            place = (s[0].flatten(1)
                     + s[1].transpose(2, 3).flatten(1)
                     + s[2].permute(0, 2, 3, 1).flatten(1)
                     + s[3].flip(2).flatten(1)
                     + s[4].flip(3).flatten(1)
                     + s[5].permute(0, 3, 2, 1).flatten(1))
            logits = torch.cat([place, d], dim=1) + torch.cat([self.done2(g), s[6].flatten(1)], dim=1)
        else:
            place = self.fallback(F.relu(self.cell(h)).flatten(1))
            logits = torch.cat([place, d], dim=1)
        return logits, self.value_mlp(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
