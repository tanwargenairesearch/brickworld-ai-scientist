import torch
import torch.nn as nn
import torch.nn.functional as F

C = 24
P = 16
H = 12

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "gae_lambda": 0.95}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.T = len(self.fps)
        self.stem = nn.Conv2d(16, C, 3, padding=1)
        self.block = nn.Conv2d(C, C, 3, padding=1)
        self.mix = nn.Conv2d(2 * C, C, 1)
        self.proj = nn.Conv2d(C, P, 1)
        self.type_hidden = nn.ModuleList([nn.Conv2d(2 * P, H, 1) for _ in self.fps])
        self.type_out = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * C, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        v = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        vp = F.pad(v, (0, 0, 0, 0, 1, 1))
        nb = torch.cat([vp[:, :, 0:Z], vp[:, :, 1:Z + 1], vp[:, :, 2:Z + 2]], dim=1)
        g = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1)
        zc = zc.view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        x = torch.cat([nb, g, zc], dim=1).permute(0, 2, 1, 3, 4).reshape(B * Z, 16, Y, X)
        f = F.relu(self.stem(x))
        f = F.relu(f + self.block(f))
        f5 = f.view(B, Z, C, Y, X)
        m = self.mix(torch.cat([f5.amax(1), f5.mean(1)], dim=1))
        f5 = F.relu(f5 + m.unsqueeze(1))
        f = f5.reshape(B * Z, C, Y, X)
        p = F.relu(self.proj(f))
        outs = []
        for fp, hid, out in zip(self.fps, self.type_hidden, self.type_out):
            dy, dx = fp
            pp = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(pp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-pp, (dy, dx), stride=1)
            h = F.relu(hid(torch.cat([mx, mn], dim=1)))
            outs.append(out(h))
        lz = torch.cat(outs, dim=1).view(B, Z, self.T, Y, X)
        place = torch.logsumexp(lz, dim=1).flatten(1)
        gfeat = torch.cat([f5.mean((1, 3, 4)), f5.amax((1, 3, 4))], dim=1)
        gg = self.glob(gfeat)
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
