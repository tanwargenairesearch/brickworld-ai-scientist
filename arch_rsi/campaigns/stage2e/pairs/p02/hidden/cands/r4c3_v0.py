import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
PW = 32
HID = 32
HEADS = 4

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.conv1 = nn.Conv2d(spec.in_channels + 2, WIDTH, 3, padding=1)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.ln1 = nn.LayerNorm(WIDTH)
        self.attn = nn.MultiheadAttention(WIDTH, HEADS, batch_first=True)
        self.ln2 = nn.LayerNorm(WIDTH)
        self.ff = nn.Sequential(nn.Linear(WIDTH, 2 * WIDTH), nn.ReLU(), nn.Linear(2 * WIDTH, WIDTH))
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
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        x = torch.cat([obs, yy, xx], dim=1)
        f = F.relu(self.conv1(x))
        f = F.relu(self.conv2(f))
        t = f.flatten(2).transpose(1, 2)
        h = self.ln1(t)
        a = self.attn(h, h, h, need_weights=False)[0]
        t = t + a
        t = t + self.ff(self.ln2(t))
        f = t.transpose(1, 2).reshape(B, WIDTH, Y, X)
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
