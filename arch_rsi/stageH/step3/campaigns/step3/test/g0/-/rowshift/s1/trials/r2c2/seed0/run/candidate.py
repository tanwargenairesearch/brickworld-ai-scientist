import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ent_coef": 0.005}

COL1 = 96
COL2 = 64
PROJ = 48
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.col = nn.Sequential(
            nn.Conv2d(cin, COL1, 1), nn.ReLU(),
            nn.Conv2d(COL1, COL2, 1), nn.ReLU())
        self.proj = nn.Conv2d(COL2, PROJ, 1)
        convs = []
        for dy, dx in self.fps:
            k = 2 * max(dy, dx) - 1
            convs.append(nn.Conv2d(PROJ, HID, k, padding=k // 2))
        self.convs = nn.ModuleList(convs)
        self.mids = nn.ModuleList([nn.Conv2d(HID, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * COL2, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        c = self.col(obs)
        p = F.relu(self.proj(c))
        res = []
        for conv, mid, out in zip(self.convs, self.mids, self.outs):
            h = F.relu(conv(p))
            h = F.relu(mid(h))
            res.append(out(h).flatten(1))
        place = torch.cat(res, dim=1)
        g = self.glob(torch.cat([c.mean((2, 3)), c.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
