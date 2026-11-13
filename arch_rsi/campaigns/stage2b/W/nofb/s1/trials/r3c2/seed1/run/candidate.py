import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 1}

HID = 32
CUM = 8
CTX = 24
POOL = 16
HEAD = 24


def fp_max(x, dy, dx):
    if dy == 1 and dx == 1:
        return x
    x = F.pad(x, (0, dx - 1, 0, dy - 1))
    return F.max_pool2d(x, (dy, dx), stride=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        T = spec.n_types
        self.T = T
        self.inp = nn.Conv2d(16, HID, 1)
        self.ctx = nn.Conv2d(HID, CTX, 3, padding=1)
        fd = HID + 2 * CUM + CTX
        self.fd = fd
        self.pre = nn.Conv2d(fd, 2 * POOL, 1)
        self.head1 = nn.Conv2d(T * 2 * POOL, T * HEAD, 1, groups=T)
        self.head2 = nn.Conv2d(T * HEAD, T, 1, groups=T)
        self.glob = nn.Sequential(nn.Linear(2 * fd, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        T = self.T
        s = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        sp = F.pad(s, (0, 0, 0, 0, 1, 1))
        below = sp[:, :, 0:Z]
        cur = sp[:, :, 1:Z + 1]
        above = sp[:, :, 2:Z + 2]
        g3 = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1)
        zc = zc.view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        u = torch.cat([below, cur, above, g3, zc], dim=1)
        u = u.permute(0, 2, 1, 3, 4).reshape(B * Z, 16, Y, X)
        h = F.relu(self.inp(u))
        h5 = h.reshape(B, Z, HID, Y, X)
        up = torch.cummax(h5[:, :, :CUM], dim=1).values
        dn = torch.cummax(h5[:, :, CUM:2 * CUM].flip(1), dim=1).values.flip(1)
        colf = h5.amax(1)
        cx = F.relu(self.ctx(colf)).unsqueeze(1).expand(B, Z, CTX, Y, X)
        f5 = torch.cat([h5, up, dn, cx], dim=2)
        f = f5.reshape(B * Z, self.fd, Y, X)
        p = self.pre(f)
        a = p[:, :POOL]
        b = p[:, POOL:]
        feats = []
        for (dy, dx) in self.fps:
            feats.append(fp_max(a, dy, dx))
            feats.append(-fp_max(-b, dy, dx))
        zf = torch.cat(feats, dim=1)
        sc = self.head2(F.relu(self.head1(zf)))
        place = torch.logsumexp(sc.reshape(B, Z, T, Y, X), dim=1).flatten(1)
        gf = torch.cat([f5.mean((1, 3, 4)), f5.amax(dim=(1, 3, 4))], dim=1)
        g = self.glob(gf)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
