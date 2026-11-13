import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LAYER_W = 32
WIDTH = 64
HID = 32
K = 16

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2}


def fp_pool(h, dy, dx):
    hp = F.pad(h, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(hp, kernel_size=(dy, dx), stride=1)
    mn = -F.max_pool2d(-hp, kernel_size=(dy, dx), stride=1)
    return torch.cat([mx, mn], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.footprints = [(int(a), int(b)) for (a, b) in spec.footprints]
        n = len(self.footprints)
        self.layer_conv = nn.Conv2d(8, LAYER_W, 3, padding=1)
        self.layer_key = nn.Conv2d(LAYER_W, K, 1)
        self.mix = nn.Conv2d(self.Z * LAYER_W, WIDTH, 1)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.g2c = nn.Linear(WIDTH, WIDTH)
        self.proj = nn.ModuleList([nn.Conv2d(WIDTH, HID, 1) for _ in range(n)])
        self.query = nn.ModuleList([nn.Conv2d(2 * HID, 2 * K, 1) for _ in range(n)])
        self.zbias = nn.Parameter(torch.zeros(n, self.Z))
        self.headmix = nn.ModuleList([nn.Conv2d(2 * HID + 2 * K, HID, 1) for _ in range(n)])
        self.out = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in range(n)])
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)
        self.scale = 1.0 / math.sqrt(2 * K)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        per = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], dim=2)
        ext = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1)
        zc = zc.view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([per, ext, zc], dim=2).reshape(B * Z, 8, Y, X)
        e = F.relu(self.layer_conv(x))
        k = self.layer_key(e)
        f = F.relu(self.mix(e.reshape(B, Z * LAYER_W, Y, X)))
        f = F.relu(self.conv2(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        fc = f + self.g2c(g)[:, :, None, None]
        outs = []
        for i, (dy, dx) in enumerate(self.footprints):
            pf = fp_pool(self.proj[i](fc), dy, dx)
            pk = fp_pool(k, dy, dx).reshape(B, Z, 2 * K, Y, X)
            q = self.query[i](F.relu(pf))
            s = (pk * q.unsqueeze(1)).sum(2) * self.scale + self.zbias[i].view(1, Z, 1, 1)
            w = torch.softmax(s, dim=1)
            a = (w.unsqueeze(2) * pk).sum(1)
            h = F.relu(self.headmix[i](torch.cat([pf, a], dim=1)))
            outs.append(self.out[i](h).flatten(1))
        place = torch.cat(outs, dim=1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
