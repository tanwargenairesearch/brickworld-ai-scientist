import torch
import torch.nn as nn
import torch.nn.functional as F

EMB = 48
LAT = 32
MID = 16
PROJ = 16
HID = 16

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = 12 + spec.in_channels + self.Z
        self.lay1 = nn.Conv2d(cin, EMB, 1)
        self.lay2 = nn.Conv2d(EMB, LAT, 1)
        self.down = nn.Conv2d(LAT, MID, 1)
        self.sp = nn.Conv2d(MID, MID, 3, padding=1)
        self.up = nn.Conv2d(MID, LAT, 1)
        self.proj = nn.Conv2d(LAT, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * LAT, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        per = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        pz = F.pad(per, (0, 0, 0, 0, 1, 1))
        local = torch.cat([pz[:, :, 0:Z], pz[:, :, 1:Z + 1], pz[:, :, 2:Z + 2]], dim=1)
        col = obs.unsqueeze(2).expand(B, C, Z, Y, X)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype).reshape(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        tok = torch.cat([local, col, eye], dim=1)
        tok = tok.permute(0, 2, 1, 3, 4).reshape(B * Z, -1, Y, X)
        h = F.relu(self.lay1(tok))
        h = F.relu(self.lay2(h))
        s = F.relu(self.sp(F.relu(self.down(h))))
        h = F.relu(h + self.up(s))
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
            u = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(u))
        lay_logits = torch.cat(res, dim=1).reshape(B, Z, -1)
        place = torch.logsumexp(lay_logits, dim=1)
        hz = h.reshape(B, Z, LAT, Y, X)
        g = self.glob(torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
