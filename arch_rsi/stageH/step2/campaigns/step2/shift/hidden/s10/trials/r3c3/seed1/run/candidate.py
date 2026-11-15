import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.001, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.01}

W = 24
PROJ = 12
HID = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = 12 + 3 + 1
        self.inp = nn.Conv2d(cin, W, 1)
        self.col = nn.Conv2d(W, W, 1)
        self.conv1 = nn.Conv2d(W, W, 3, padding=1)
        self.ctx = nn.Linear(2 * W, W)
        self.conv2 = nn.Conv2d(W, W, 3, padding=1)
        self.proj = nn.Conv2d(W, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        Z = self.Z
        occ = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        occ = F.pad(occ, (0, 0, 0, 0, 1, 1))
        nb = torch.cat([occ[:, :, 0:Z], occ[:, :, 1:Z + 1], occ[:, :, 2:Z + 2]], dim=1)
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zp = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        x = torch.cat([nb, gl, zp], dim=1).permute(0, 2, 1, 3, 4).reshape(B * Z, 16, Y, X)

        h0 = F.relu(self.inp(x))
        hv = h0.view(B, Z, W, Y, X)
        c = self.col(hv.amax(1))
        h = F.relu(hv + c.unsqueeze(1)).reshape(B * Z, W, Y, X)
        h = F.relu(h + self.conv1(h))
        hv = h.view(B, Z, W, Y, X)
        g = self.ctx(torch.cat([hv.mean((1, 3, 4)), hv.amax((1, 3, 4))], dim=1))
        h = (hv + g[:, None, :, None, None]).reshape(B * Z, W, Y, X)
        h = F.relu(h + self.conv2(h))

        p = self.proj(h)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            t = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(t))
        s = torch.cat(res, dim=1).reshape(B, Z, -1)
        place = torch.logsumexp(s, dim=1)

        hv2 = h.view(B, Z, W, Y, X)
        gg = self.glob(torch.cat([hv2.mean((1, 3, 4)), hv2.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
