import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
W = 48
HID = 64

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "lr": 0.001, "ent_coef": 0.01}


def apply_t(m, k):
    if k >= 4:
        m = m.transpose(-1, -2)
    r = k % 4
    if r == 1 or r == 3:
        m = torch.flip(m, dims=[-1])
    if r == 2 or r == 3:
        m = torch.flip(m, dims=[-2])
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.nh = 8 if int(spec.Y) == int(spec.X) else 4
        self.stem1 = nn.Conv2d(spec.in_channels, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, W, 1)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.hid = nn.Conv2d(2 * W, HID, 1)
        self.out = nn.Conv2d(HID, self.nh * 2 * self.T, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        T = self.T
        s = F.relu(self.stem2(F.relu(self.stem1(obs))))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        h = F.relu(self.hid(torch.cat([s, f], dim=1)))
        o = self.out(h)
        place = None
        for k in range(self.nh):
            m = apply_t(o[:, k * 2 * T:(k + 1) * 2 * T], k)
            tm = m[:, :T].flatten(1)
            cm = m[:, T:].permute(0, 2, 3, 1).flatten(1)
            p = tm + cm
            place = p if place is None else place + p
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
