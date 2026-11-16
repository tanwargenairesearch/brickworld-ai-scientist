import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0005, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 4, "ent_coef": 0.01, "gamma": 0.98, "gae_lambda": 0.95}

HID = 40
PROJ = 16
MIX = 24


def fp_pool(t, dy, dx):
    if dy * dx == 1:
        return t, t, t
    q = F.pad(t, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = 12 + spec.in_channels + self.z
        self.enc = nn.Sequential(
            nn.Conv2d(cin, HID, 1), nn.ReLU(),
            nn.Conv2d(HID, HID, 1), nn.ReLU())
        self.proj = nn.Conv2d(HID, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, MIX, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(MIX, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * HID, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def voxels(self, obs):
        b, c, yy, xx = obs.shape
        z = self.z
        vox = obs[:, :4 * z].reshape(b, 4, z, yy, xx)
        pad = F.pad(vox, (0, 0, 0, 0, 1, 1))
        loc = torch.cat([vox, pad[:, :, :z], pad[:, :, 2:]], dim=1).permute(0, 2, 1, 3, 4)
        col = obs.unsqueeze(1).expand(b, z, c, yy, xx)
        zid = torch.eye(z, device=obs.device, dtype=obs.dtype).reshape(1, z, z, 1, 1).expand(b, z, z, yy, xx)
        return torch.cat([loc, col, zid], dim=2).reshape(b * z, 12 + c + z, yy, xx)

    def forward(self, obs, mask):
        b, _, yy, xx = obs.shape
        z = self.z
        h = self.enc(self.voxels(obs))
        p = self.proj(h)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            mx, mn, av = fp_pool(p, dy, dx)
            s = out(F.relu(mix(torch.cat([mx, mn, av], dim=1))))
            s = s.reshape(b, z, yy * xx)
            res.append(torch.logsumexp(s, dim=1))
        place = torch.cat(res, dim=1)
        hv = h.reshape(b, z, HID, yy, xx)
        g = self.glob(torch.cat([hv.mean((1, 3, 4)), hv.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
