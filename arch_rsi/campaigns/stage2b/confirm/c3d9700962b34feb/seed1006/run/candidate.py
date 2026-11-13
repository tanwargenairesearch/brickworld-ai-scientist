"""r2c3 per-layer stream with 2 PPO epochs for about 1.86x the samples."""
import torch
import torch.nn as nn
import torch.nn.functional as F

W = 32
P = 16
HID = 16

RECIPE = {"ppo_epochs": 2, "n_minibatches": 16}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.conv_g = nn.Conv2d(spec.in_channels, W, 3, padding=1)
        self.conv_l = nn.Conv2d(8 + self.Z, W, 3, padding=1)
        self.mix = nn.Conv2d(W, W, 1)
        self.proj = nn.Conv2d(W, P, 1)
        self.type_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * P, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def pool(self, f, dy, dx):
        if dy == 1 and dx == 1:
            return f, f
        p = F.pad(f, (0, dx - 1, 0, dy - 1))
        return F.max_pool2d(p, (dy, dx), stride=1), F.avg_pool2d(p, (dy, dx), stride=1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        cur = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        below = F.pad(cur, (0, 0, 0, 0, 1, 0))[:, :, :Z]
        zoh = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        lay = torch.cat([cur, below, zoh], dim=1)
        lay = lay.permute(0, 2, 1, 3, 4).reshape(B * Z, 8 + Z, Y, X)
        g2d = self.conv_g(obs)
        g2d = g2d.unsqueeze(1).expand(B, Z, W, Y, X).reshape(B * Z, W, Y, X)
        h = F.relu(g2d + self.conv_l(lay))
        h = F.relu(self.mix(h))
        hp = self.proj(h)
        outs = []
        for (dy, dx), head in zip(self.fps, self.type_heads):
            pmax, pmean = self.pool(hp, dy, dx)
            s = head(torch.cat([hp, pmax, pmean], dim=1)).view(B, Z, Y * X)
            outs.append(torch.logsumexp(s, dim=1))
        place = torch.cat(outs, dim=1)
        h5 = h.view(B, Z, W, Y, X)
        gfeat = torch.cat([h5.mean((1, 3, 4)), h5.amax((1, 3, 4))], dim=1)
        g = self.glob(gfeat)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
