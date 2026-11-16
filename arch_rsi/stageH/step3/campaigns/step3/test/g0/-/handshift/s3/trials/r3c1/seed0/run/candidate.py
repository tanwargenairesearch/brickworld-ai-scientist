import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.001}

STEM = 64
W = 48
R = 8
H = 256
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        self.nplace = int(spec.n_place)
        Y = int(spec.Y)
        X = int(spec.X)
        cin = int(spec.in_channels) + self.nt
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU())
        self.red = nn.Conv2d(W, R, 1)
        self.fc = nn.Linear(R * Y * X + G, H)
        self.out = nn.Linear(H, self.nplace)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        m = mask[:, :self.nplace].to(obs.dtype).reshape(B, self.nt, Y, X)
        s = self.stem(torch.cat([obs, m], dim=1))
        x = F.relu(s + self.c1(s))
        f = F.relu(x + self.c2(x))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        r = F.relu(self.red(f)).flatten(1)
        h = F.relu(self.fc(torch.cat([r, g], dim=1)))
        place = self.out(h)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
