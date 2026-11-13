import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 20
HID = 12

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "ppo_epochs": 2,
          "lr": 0.002, "gamma": 0.99, "gae_lambda": 0.95}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        c_in = 16
        self.body = nn.Sequential(
            nn.Conv2d(c_in, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.proj = nn.Sequential(nn.Conv2d(WIDTH, HID, 1), nn.ReLU())
        self.type_hidden = nn.ModuleList(
            [nn.Conv2d(WIDTH + 2 * HID, HID, 1) for _ in self.fps])
        self.type_out = nn.ModuleList([nn.Conv2d(HID, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        gch = obs[:, 4 * Z:4 * Z + 3]
        per = torch.stack([b, t, m, e], dim=2)
        ones = torch.ones(B, 1, 2, Y, X, device=obs.device, dtype=obs.dtype)
        zeros2 = torch.zeros(B, 1, 2, Y, X, device=obs.device, dtype=obs.dtype)
        ground = torch.cat([ones, zeros2], dim=2)
        sky = torch.zeros(B, 1, 4, Y, X, device=obs.device, dtype=obs.dtype)
        below = torch.cat([ground, per[:, :-1]], dim=1)
        above = torch.cat([per[:, 1:], sky], dim=1)
        gexp = gch[:, None].expand(B, Z, 3, Y, X)
        zc = torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1)
        zc = zc.view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([below, per, above, gexp, zc], dim=2).reshape(B * Z, 16, Y, X)
        f = self.body(x)
        p = self.proj(f)
        outs = []
        for i, (dy, dx) in enumerate(self.fps):
            pp = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(pp, (dy, dx), stride=1)
            av = F.avg_pool2d(pp, (dy, dx), stride=1)
            h = F.relu(self.type_hidden[i](torch.cat([f, mx, av], dim=1)))
            o = self.type_out[i](h).reshape(B, Z, 2, Y, X)
            s = o[:, :, 0]
            w = torch.softmax(o[:, :, 1], dim=1)
            outs.append((w * s).sum(1))
        place = torch.stack(outs, dim=1).flatten(1)
        f5 = f.reshape(B, Z, WIDTH, Y, X)
        pooled = torch.cat([f5.mean((1, 3, 4)), f5.amax((1, 3, 4))], dim=1)
        gl = self.glob(pooled)
        logits = torch.cat([place, self.done_head(gl)], dim=1)
        return logits, self.value_head(gl).squeeze(1)


def build_policy(spec):
    return Net(spec)
