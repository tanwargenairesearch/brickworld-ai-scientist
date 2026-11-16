import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "lr": 0.001, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

STEM = 64
W = 40
HID = 32
NMAP = 6


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.c3 = nn.Conv2d(W, W, 3, padding=1)
        self.hconv = nn.Conv2d(W, HID, 3, padding=1)
        self.outs = nn.ModuleList([nn.Conv2d(HID, self.T, 1) for _ in range(NMAP)])
        for o in self.outs:
            nn.init.orthogonal_(o.weight, gain=0.01)
            nn.init.zeros_(o.bias)
        self.gate = nn.Parameter(torch.zeros(NMAP))
        self.glob = nn.Sequential(nn.Linear(2 * W + HID, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        s = self.stem(obs)
        x = F.relu(s + self.c1(s))
        x = F.relu(x + self.c2(x))
        x = F.relu(x + self.c3(x))
        h = F.relu(self.hconv(x))
        maps = []
        for k, o in enumerate(self.outs):
            m = o(h)
            if k == 0:
                v = m
            elif k == 1:
                v = m.transpose(2, 3)
            elif k == 2:
                v = m.flip(2)
            elif k == 3:
                v = m.flip(3)
            elif k == 4:
                v = m.permute(0, 2, 3, 1)
            else:
                v = m.permute(0, 3, 2, 1)
            maps.append(v.reshape(B, -1))
        w = torch.softmax(self.gate, 0) * NMAP
        place = (torch.stack(maps, 1) * w.view(1, -1, 1)).sum(1)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
