"""Layer-shared 2D conv over (layer z, layer below, layer above) inputs, attention pooling over z,
then a 2D mixing conv with placement, DONE and value heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F

LW = 24
HW = 48

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "ent_coef": 0.005}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        cin = 7 + 3 + 1
        self.l1 = nn.Conv2d(cin, LW, 3, padding=1)
        self.l2 = nn.Conv2d(LW, LW, 3, padding=1)
        self.score = nn.Conv2d(LW, 1, 1)
        self.mix = nn.Conv2d(2 * LW, HW, 3, padding=1)
        self.place_head = nn.Conv2d(HW, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * HW, HW), nn.ReLU())
        self.done_head = nn.Linear(HW, 1)
        self.value_head = nn.Linear(HW, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        build = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        bb = F.pad(build, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        bt = F.pad(tgt, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        at = F.pad(tgt, (0, 0, 0, 0, 0, 1), value=0.0)[:, 1:]
        per = torch.stack([build, tgt, miss, extra, bb, bt, at], dim=2)
        g = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([per, g, zc], dim=2).reshape(B * Z, 11, Y, X)
        f = F.relu(self.l1(x))
        f = F.relu(f + self.l2(f))
        s = self.score(f).view(B, Z, 1, Y, X)
        a = torch.softmax(s, dim=1)
        f = f.view(B, Z, LW, Y, X)
        pooled = torch.cat([(a * f).sum(1), f.amax(1)], dim=1)
        h = F.relu(self.mix(pooled))
        place = self.place_head(h).flatten(1)
        gg = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
