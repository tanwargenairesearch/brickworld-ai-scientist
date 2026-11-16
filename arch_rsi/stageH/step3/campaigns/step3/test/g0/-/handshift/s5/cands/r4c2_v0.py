import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 64, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2,
          "lr": 0.0005, "ent_coef": 0.003, "gamma": 0.97, "gae_lambda": 0.95}

W = 40
HID = 32
G = 64
NL = 4


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.npl = int(spec.n_place)
        self.stem = nn.Conv2d(int(spec.in_channels) + self.nt, W, 1)
        self.convs = nn.ModuleList([nn.Conv2d(W, W, 3, padding=1) for _ in range(NL)])
        self.h1 = nn.Conv2d(W, HID, 1)
        self.h2 = nn.Conv2d(HID, self.nt, 1)
        nn.init.normal_(self.h2.weight, std=0.01)
        nn.init.zeros_(self.h2.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.npl].to(obs.dtype).reshape(B, self.nt, self.Y, self.X)
        x = F.relu(self.stem(torch.cat([obs, m], dim=1)))
        for c in self.convs:
            x = F.relu(x + c(x))
        place = self.h2(F.relu(self.h1(x))).flatten(1)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
