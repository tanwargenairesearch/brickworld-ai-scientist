import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.0015, "ent_coef": 0.02}

STEM = 64
COL = 48
WIDTH = 40
HID = 24
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        cin = spec.in_channels + 2 * self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.ha = nn.Sequential(nn.Conv2d(WIDTH, HID, 3, padding=1), nn.ReLU(), nn.Conv2d(HID, self.T, 1))
        self.hb = nn.Sequential(nn.Conv2d(WIDTH, HID, 3, padding=1), nn.ReLU(), nn.Conv2d(HID, self.T, 1))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        T = self.T
        m = mask[:, :self.n_place].to(obs.dtype)
        ma = m.reshape(B, T, Y, X)
        mb = m.reshape(B, T, X, Y).transpose(2, 3)
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        s = self.stem(torch.cat([obs, ma, mb, ys, xs], dim=1))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        la = self.ha(f).flatten(1)
        lb = self.hb(f.transpose(2, 3).contiguous()).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([la + lb, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
