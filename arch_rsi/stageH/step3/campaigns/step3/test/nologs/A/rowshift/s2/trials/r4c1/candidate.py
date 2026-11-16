import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.003, "lr": 0.001, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32}

W = 32
HID = 48
GH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = int(spec.in_channels)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Conv2d(cin, W, 1)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.ctx = nn.Linear(2 * W, W)
        self.fp = nn.ModuleList([nn.Conv2d(cin + W, HID, (dy, dx)) for dy, dx in self.fps])
        self.mid = nn.ModuleList([nn.Conv2d(HID, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        x = F.relu(self.stem(obs))
        x = F.relu(x + self.c1(x))
        c = torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)
        x = x + self.ctx(c)[:, :, None, None]
        f = F.relu(x + self.c2(x))
        z = torch.cat([obs, f], dim=1)
        res = []
        for (dy, dx), fp, mid, out in zip(self.fps, self.fp, self.mid, self.outs):
            q = F.pad(z, (0, dx - 1, 0, dy - 1))
            h = F.relu(fp(q))
            h = F.relu(h + mid(h))
            res.append(out(h))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
