import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
LH = 32
LF = 16
LP = 8
LM = 16

RECIPE = {"ppo_epochs": 2, "n_minibatches": 16, "ent_coef": 0.005}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        lin = 4 * (2 * Z - 1) + 3 + Z
        self.l1 = nn.Conv2d(lin, LH, 1)
        self.l2 = nn.Conv2d(LH, LF, 1)
        self.lctx = nn.Conv2d(WIDTH, LF, 1)
        self.lproj = nn.Conv2d(LF, LP, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * LP, LM, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(LM, 1, 1) for _ in self.fps])
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.ctx = nn.Linear(2 * WIDTH, WIDTH)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + LF, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def layer_input(self, obs):
        B, C, Y, X = obs.shape
        Z = self.Z
        st = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        st = F.pad(st, (0, 0, 0, 0, Z - 1, Z - 1))
        rel = torch.stack([st[:, :, z:z + 2 * Z - 1] for z in range(Z)], dim=1)
        rel = rel.reshape(B, Z, 4 * (2 * Z - 1), Y, X)
        rest = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype)
        oh = eye[None, :, :, None, None].expand(B, Z, Z, Y, X)
        inp = torch.cat([rel, rest, oh], dim=2)
        return inp.reshape(B * Z, inp.shape[2], Y, X)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        pooled = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        f2 = F.relu(f + self.ctx(pooled)[:, :, None, None])
        c = self.lctx(f2)
        c = c.unsqueeze(1).expand(B, Z, LF, Y, X).reshape(B * Z, LF, Y, X)
        h = F.relu(self.l1(self.layer_input(obs)))
        h = F.relu(self.l2(h) + c)
        p = self.lproj(h)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            res.append(out(F.relu(mix(torch.cat([mx, mn, av], dim=1)))))
        sc = torch.cat(res, dim=1)
        T = sc.shape[1]
        place = sc.reshape(B, Z, T, Y, X).sum(1).flatten(1)
        hl = h.reshape(B, Z, LF, Y, X).mean((1, 3, 4))
        g = self.glob(torch.cat([f2.mean((2, 3)), f2.amax((2, 3)), hl], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
