import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
RES_SCALE = 0.3


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.fps = [(int(f[0]), int(f[1])) for f in spec.footprints]
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.res_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        nn.init.zeros_(self.res_head.weight)
        nn.init.zeros_(self.res_head.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + 2, WIDTH), nn.ReLU())
        self.done_res = nn.Linear(WIDTH, 1)
        nn.init.zeros_(self.done_res.weight)
        nn.init.zeros_(self.done_res.bias)
        self.value_head = nn.Linear(WIDTH, 1)
        # correct, overhang bonus, wrong, buried, layer
        self.w = nn.Parameter(torch.tensor([2.0, 2.0, 6.0, 6.0, 0.2]))
        self.done_w = nn.Parameter(torch.tensor([8.0, -4.0]))

    def prior(self, obs):
        Z = self.Z
        h = torch.round(obs[:, 4 * Z] * Z).unsqueeze(1)
        miss = obs[:, 2 * Z:3 * Z]
        zs = torch.arange(Z, device=obs.device, dtype=obs.dtype).view(1, Z, 1, 1)
        heq = (h == zs).to(obs.dtype)
        w = self.w
        outs = []
        for (dy, dx) in self.fps:
            n = float(dy * dx)
            L = F.max_pool2d(h, (dy, dx), stride=1)
            mc = torch.round(F.avg_pool2d(miss, (dy, dx), stride=1) * n)
            hc = torch.round(F.avg_pool2d(heq, (dy, dx), stride=1) * n)
            onehot = (L == zs).to(obs.dtype)
            correct = (onehot * mc).sum(1)
            below = torch.cumsum(mc, dim=1) - mc
            buried = (onehot * below).sum(1)
            at_l = (onehot * hc).sum(1)
            over = n - at_l
            wrong = n - correct
            perfect = ((wrong < 0.5) & (buried < 0.5)).to(obs.dtype)
            s = (w[0] * correct + w[1] * over * perfect - w[2] * wrong
                 - w[3] * buried - w[4] * L.squeeze(1))
            s = F.pad(s, (0, dx - 1, 0, dy - 1))
            outs.append(s)
        return torch.stack(outs, dim=1)

    def forward(self, obs, mask):
        Z = self.Z
        B = obs.shape[0]
        f = self.body(obs)
        place = self.prior(obs) + RES_SCALE * self.res_head(f)
        place = place.flatten(1)
        miss_tot = obs[:, 2 * Z:3 * Z].sum((1, 2, 3))
        targ_tot = obs[:, Z:2 * Z].sum((1, 2, 3)).clamp(min=1.0)
        nomiss = (miss_tot < 0.5).to(obs.dtype)
        extra = torch.stack([miss_tot / targ_tot, obs[:, 4 * Z + 2].mean((1, 2))], dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), extra], dim=1))
        done = self.done_w[0] * nomiss + self.done_w[1] + RES_SCALE * self.done_res(g).squeeze(1)
        logits = torch.cat([place, done.view(B, 1)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
