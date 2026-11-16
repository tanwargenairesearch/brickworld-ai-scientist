import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.0015, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

STEM = 64
COL = 48
WIDTH = 40
HID = 48


def dihedral(m, k):
    if k >= 4:
        m = m.transpose(2, 3)
    if k % 2 == 1:
        m = m.flip([2])
    if (k // 2) % 2 == 1:
        m = m.flip([3])
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.K = 8 if self.Y == self.X else 4
        cin = int(spec.in_channels) + 2 * self.T
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hconv = nn.Conv2d(WIDTH, HID, 3, padding=1)
        self.out = nn.Conv2d(HID, 2 * self.K * self.T, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        T, Y, X = self.T, self.Y, self.X
        mp = mask[:, :self.n_place].to(obs.dtype)
        m1 = mp.reshape(B, T, Y, X)
        m2 = mp.reshape(B, Y, X, T).permute(0, 3, 1, 2)
        s = self.stem(torch.cat([obs, m1, m2], dim=1))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        h = F.relu(self.hconv(f))
        o = self.out(h).reshape(B, 2, self.K, T, Y, X)
        pa = dihedral(o[:, 0, 0], 0)
        pb = dihedral(o[:, 1, 0], 0)
        for k in range(1, self.K):
            pa = pa + dihedral(o[:, 0, k], k)
            pb = pb + dihedral(o[:, 1, k], k)
        place = pa.flatten(1) + pb.permute(0, 2, 3, 1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
