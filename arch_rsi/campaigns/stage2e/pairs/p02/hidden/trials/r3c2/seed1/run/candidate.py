import torch
import torch.nn as nn
import torch.nn.functional as F

LW = 32
WIDTH = 64
PW = 32
HID = 32

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.l1 = nn.Conv2d(16, LW, 1)
        self.l2 = nn.Conv2d(LW, LW, 3, padding=1)
        self.att = nn.Conv2d(LW, 1, 1)
        self.body = nn.Conv2d(2 * LW, WIDTH, 3, padding=1)
        self.proj = nn.Conv2d(WIDTH, PW, 1)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.fp_convs = nn.ModuleList([nn.Conv2d(PW, HID, (dy, dx)) for (dy, dx) in self.fps])
        self.pool_convs = nn.ModuleList([nn.Conv2d(2 * PW, HID, 1) for _ in self.fps])
        self.out_convs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        per = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], dim=2)
        below = F.pad(per, (0, 0, 0, 0, 0, 0, 1, 0))[:, :Z]
        above = F.pad(per, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:]
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([per, below, above, gl, zc], dim=2).reshape(B * Z, 16, Y, X)
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        w = torch.softmax(self.att(h).reshape(B, Z, 1, Y, X), dim=1)
        h = h.reshape(B, Z, LW, Y, X)
        sg = (w * h).sum(1)
        mxz = h.amax(1)
        f = F.relu(self.body(torch.cat([sg, mxz], dim=1)))
        p = self.proj(f)
        outs = []
        for i, (dy, dx) in enumerate(self.fps):
            pp = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(pp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-pp, (dy, dx), stride=1)
            hh = F.relu(self.fp_convs[i](pp) + self.pool_convs[i](torch.cat([mx, mn], dim=1)))
            outs.append(self.out_convs[i](hh))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
