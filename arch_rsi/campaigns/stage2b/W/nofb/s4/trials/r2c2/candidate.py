import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
HID = 24
COL = 64
RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 3, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        W = WIDTH
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Conv2d(spec.in_channels, W, 3, padding=1)
        self.col1 = nn.Conv2d(spec.in_channels, COL, 1)
        self.col2 = nn.Conv2d(COL, W, 1)
        self.conv1 = nn.Conv2d(W, W, 3, padding=1)
        self.conv2 = nn.Conv2d(W, W, 3, padding=1)
        self.glob_in = nn.Linear(2 * W, W)
        self.glob_out = nn.Linear(W, W)
        self.fp_proj = nn.Conv2d(W, HID, 1)
        self.fp_convs = nn.ModuleList([nn.Conv2d(W, HID, (dy, dx)) for dy, dx in self.footprints])
        self.fp_out = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.footprints])
        for m in self.fp_out:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
        self.trunk = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        h = F.relu(self.stem(obs) + self.col2(F.relu(self.col1(obs))))
        h = h + F.relu(self.conv1(h))
        g = F.relu(self.glob_in(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)))
        h = h + self.glob_out(g)[:, :, None, None]
        h = F.relu(h + F.relu(self.conv2(h)))
        p = self.fp_proj(h)
        outs = []
        for i, (dy, dx) in enumerate(self.footprints):
            hp = F.pad(h, (0, dx - 1, 0, dy - 1))
            a = self.fp_convs[i](hp)
            pp = F.pad(p, (0, dx - 1, 0, dy - 1))
            m = F.max_pool2d(pp, (dy, dx), stride=1)
            outs.append(self.fp_out[i](F.relu(a + m)))
        place = torch.cat(outs, dim=1).flatten(1)
        t = self.trunk(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(t)], dim=1)
        return logits, self.value_head(t).squeeze(1)


def build_policy(spec):
    return Net(spec)
