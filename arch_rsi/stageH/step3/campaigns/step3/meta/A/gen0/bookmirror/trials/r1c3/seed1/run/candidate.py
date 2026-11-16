import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16}

STEM = 64
TR = 32
CE = 16
LW = 32
PROJ = 16
HID = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, TR, 1), nn.ReLU())
        self.c1 = nn.Conv2d(TR, TR, 3, padding=1)
        self.c2 = nn.Conv2d(TR, TR, 3, padding=1)
        self.col = nn.Conv2d(TR, CE, 1)
        self.lin = nn.Conv2d(15 + self.Z, LW, 1)
        self.l2 = nn.Conv2d(LW + CE, LW, 1)
        self.proj = nn.Conv2d(LW, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * TR, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        col = F.relu(self.col(f))

        lay = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        lp = F.pad(lay, (0, 0, 0, 0, 1, 1))
        sl = torch.cat([lp[:, :, 0:Z], lp[:, :, 1:Z + 1], lp[:, :, 2:Z + 2]], dim=1)
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zid = torch.eye(Z, device=obs.device, dtype=obs.dtype).reshape(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        inp = torch.cat([sl, gl, zid], dim=1).permute(0, 2, 1, 3, 4).reshape(B * Z, 15 + Z, Y, X)
        h = F.relu(self.lin(inp))
        cz = col.unsqueeze(1).expand(B, Z, CE, Y, X).reshape(B * Z, CE, Y, X)
        h = F.relu(self.l2(torch.cat([h, cz], dim=1)))
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
            o = out(F.relu(mix(torch.cat([mx, mn, av], dim=1))))
            o = o.reshape(B, Z, 2, Y, X)
            sc = (torch.sigmoid(o[:, :, 0]) * o[:, :, 1]).sum(1)
            res.append(sc.flatten(1))
        place = torch.cat(res, dim=1)

        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
