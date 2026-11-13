import torch
import torch.nn as nn
import torch.nn.functional as F

W = 24
H = 16
G = 64
RECIPE = {"n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "gamma": 0.97, "lr": 0.001}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.foots = [(int(f[0]), int(f[1])) for f in spec.footprints]
        self.stem = nn.Conv2d(16, W, 1)
        self.sp1 = nn.Conv2d(W, W, 3, padding=1)
        self.mix = nn.Conv2d(2 * W, W, 3, padding=1)
        self.sp2 = nn.Conv2d(W, W, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU())
        self.film = nn.Linear(G, W)
        self.pre = nn.Conv2d(W, H, 1)
        heads = []
        for _ in self.foots:
            last = nn.Conv2d(H, 1, 1)
            nn.init.orthogonal_(last.weight, 0.01)
            nn.init.zeros_(last.bias)
            heads.append(nn.Sequential(nn.Conv2d(2 * H + W, H, 1), nn.ReLU(), last))
        self.heads = nn.ModuleList(heads)
        self.done_head = nn.Linear(G, 1)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        lay = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        p = F.pad(lay, (0, 0, 0, 0, 1, 1))
        cur = p[:, :, 1:Z + 1]
        below = p[:, :, 0:Z]
        above = p[:, :, 2:Z + 2]
        gch = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        x = torch.cat([cur, below, above, gch, zc], dim=1)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * Z, 16, Y, X)
        h = F.relu(self.stem(x))
        h = F.relu(self.sp1(h) + h)
        hz = h.reshape(B, Z, W, Y, X)
        col = torch.cat([hz.mean(1), hz.amax(1)], dim=1)
        m = F.relu(self.mix(col))
        g = self.glob(torch.cat([m.mean((2, 3)), m.amax((2, 3))], dim=1))
        ctx = m + self.film(g)[:, :, None, None]
        h = h + ctx.unsqueeze(1).expand(B, Z, W, Y, X).reshape(B * Z, W, Y, X)
        h = F.relu(self.sp2(h) + h)
        pr = self.pre(h)
        outs = []
        for i, (dy, dx) in enumerate(self.foots):
            if dy == 1 and dx == 1:
                mx = pr
                mn = pr
            else:
                pp = F.pad(pr, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(pp, (dy, dx), stride=1)
                mn = -F.max_pool2d(-pp, (dy, dx), stride=1)
            outs.append(self.heads[i](torch.cat([mx, mn, h], dim=1)))
        s = torch.cat(outs, dim=1)
        T = s.shape[1]
        s = s.reshape(B, Z, T, Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
