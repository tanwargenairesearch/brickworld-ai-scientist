import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "ppo_epochs": 3, "rollout_len": 16, "n_minibatches": 32,
          "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01}

H1 = 96
E = 64
HID = 48


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.cell1 = nn.Conv2d(C, H1, 1)
        self.cell2 = nn.Conv2d(H1, E, 1)
        self.cn = nn.GroupNorm(8, E)
        self.fconv = nn.ModuleList([nn.Conv2d(E, HID, (dy, dx)) for dy, dx in self.fps])
        self.fn = nn.ModuleList([nn.GroupNorm(8, HID) for _ in self.fps])
        self.hid = nn.ModuleList([nn.Conv2d(HID, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Linear(2 * E, 64)
        self.glob2 = nn.Linear(64, 64)
        self.type_bias = nn.Linear(64, len(self.fps))
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                ortho(m, math.sqrt(2))
        for m in self.outs:
            ortho(m, 0.01)
        ortho(self.type_bias, 0.01)
        ortho(self.done_head, 0.01)
        ortho(self.value_head, 1.0)

    def forward(self, obs, mask):
        e = F.relu(self.cell1(obs))
        e = F.relu(self.cn(self.cell2(e)))
        g = F.relu(self.glob(torch.cat([e.mean((2, 3)), e.amax((2, 3))], dim=1)))
        g = F.relu(self.glob2(g))
        tb = self.type_bias(g)
        res = []
        for i, ((dy, dx), fc, fn, hd, out) in enumerate(zip(self.fps, self.fconv, self.fn, self.hid, self.outs)):
            q = F.pad(e, (0, dx - 1, 0, dy - 1))
            hh = F.relu(fn(fc(q)))
            hh = F.relu(hd(hh))
            res.append(out(hh) + tb[:, i].view(-1, 1, 1, 1))
        place = torch.cat(res, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
