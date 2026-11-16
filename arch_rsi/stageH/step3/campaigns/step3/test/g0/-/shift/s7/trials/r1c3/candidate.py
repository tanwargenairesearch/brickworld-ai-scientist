import torch
import torch.nn as nn
import torch.nn.functional as F

H = 32
P = 16
HID = 24
G = 64

RECIPE = {"n_envs": 64, "rollout_len": 32, "n_minibatches": 8, "ppo_epochs": 2}


def fp_pool(p, dy, dx):
    if dy * dx == 1:
        return torch.cat([p, p, p], dim=1)
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return torch.cat([mx, mn, av], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.col_in = nn.Conv2d(spec.in_channels, H, 1)
        self.col_ctx = nn.Conv2d(H, H, 3, padding=1)
        self.lay_in = nn.Conv2d(6, H, 1)
        self.z_emb = nn.Embedding(self.Z, H)
        self.l2 = nn.Conv2d(H, H, 1)
        self.l3 = nn.Conv2d(H, H, 1)
        self.proj = nn.Conv2d(H, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * H, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        build = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        below_b = F.pad(build, (0, 0, 0, 0, 1, 0))[:, :Z]
        below_m = F.pad(miss, (0, 0, 0, 0, 1, 0))[:, :Z]
        layer = torch.stack([build, tgt, miss, extra, below_b, below_m], dim=2)
        layer = layer.reshape(B * Z, 6, Y, X)
        c0 = self.col_in(obs)
        cctx = F.relu(self.col_ctx(F.relu(c0)))
        zi = torch.arange(Z, device=obs.device)
        ze = self.z_emb(zi).view(1, Z, H, 1, 1)
        h = self.lay_in(layer).view(B, Z, H, Y, X) + c0.unsqueeze(1) + ze
        h = F.relu(h).reshape(B * Z, H, Y, X)
        cc = cctx.unsqueeze(1).expand(B, Z, H, Y, X).reshape(B * Z, H, Y, X)
        h = F.relu(self.l2(h) + cc)
        h = F.relu(h + self.l3(h))
        p = self.proj(h)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            m = F.relu(mix(fp_pool(p, dy, dx)))
            o = out(m).view(B, Z, 2, Y, X)
            w = torch.softmax(o[:, :, 0], dim=1)
            s = (w * o[:, :, 1]).sum(1)
            res.append(s.flatten(1))
        place = torch.cat(res, dim=1)
        hz = h.view(B, Z, H, Y, X)
        g = self.glob(torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
