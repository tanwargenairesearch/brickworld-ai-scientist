import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.0015}

STEM = 64
W = 48
HID = 32
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        self.nplace = int(spec.n_place)
        cin = int(spec.in_channels) + self.nt
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU())
        self.hloc = nn.Conv2d(W, HID, 1)
        self.hglob = nn.Linear(G, HID)
        self.hout = nn.Conv2d(HID, self.nt, 3, padding=1)
        nn.init.normal_(self.hout.weight, std=0.01)
        nn.init.zeros_(self.hout.bias)
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        m = mask[:, :self.nplace].to(obs.dtype).reshape(B, self.nt, Y, X)
        s = self.stem(torch.cat([obs, m], dim=1))
        x = F.relu(s + self.c1(s))
        f = F.relu(x + self.c2(x))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = F.relu(self.hloc(f) + self.hglob(g)[:, :, None, None])
        place = self.hout(h).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
