import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 40
HID = 32

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "ppo_epochs": 2}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        W = WIDTH
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Conv2d(spec.in_channels, W, 3, padding=1)
        self.block1 = nn.Conv2d(W, W, 3, padding=1)
        self.ctx = nn.Linear(2 * W, W)
        self.block2 = nn.Conv2d(W, W, 3, padding=1)
        self.proj = nn.Sequential(nn.Conv2d(W, HID, 1), nn.ReLU())
        self.type_hidden = nn.ModuleList(
            [nn.Conv2d(W + 3 * HID, HID, 1) for _ in self.fps])
        self.type_out = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        h = F.relu(self.stem(obs))
        h = h + F.relu(self.block1(h))
        c = self.ctx(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        h = F.relu(h + c[:, :, None, None])
        f = h + F.relu(self.block2(h))
        g = self.proj(f)
        outs = []
        for i, (dy, dx) in enumerate(self.fps):
            gp = F.pad(g, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(gp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-gp, (dy, dx), stride=1)
            av = F.avg_pool2d(gp, (dy, dx), stride=1)
            t = F.relu(self.type_hidden[i](torch.cat([f, mx, mn, av], dim=1)))
            outs.append(self.type_out[i](t))
        place = torch.cat(outs, dim=1).flatten(1)
        gl = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(gl)], dim=1)
        return logits, self.value_head(gl).squeeze(1)


def build_policy(spec):
    return Net(spec)
