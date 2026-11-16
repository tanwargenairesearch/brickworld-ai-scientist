import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "lr": 0.001, "gamma": 0.95, "gae_lambda": 0.9, "ent_coef": 0.01, "n_minibatches": 16}

D = 48
P = 32
H = 64
GH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.nt = int(spec.n_types)
        self.npl = int(spec.n_place)
        self.col = nn.Sequential(
            nn.Conv2d(spec.in_channels + self.nt, 64, 1), nn.ReLU(),
            nn.Conv2d(64, D, 1))
        self.pos = nn.Parameter(torch.randn(1, D, int(spec.Y), int(spec.X)) * 0.02)
        self.loc = nn.Conv2d(D, D, 3, padding=1)
        self.enc = nn.ModuleList([
            nn.TransformerEncoderLayer(D, 4, 2 * D, dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(2)])
        self.proj = nn.Conv2d(2 * D, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(P * dy * dx, H, 1) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * D, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        m = mask[:, :self.npl].to(obs.dtype).reshape(B, self.nt, Y, X)
        c = F.relu(self.col(torch.cat([obs, m], dim=1)) + self.pos)
        h = F.relu(c + self.loc(c))
        t = h.flatten(2).transpose(1, 2)
        for layer in self.enc:
            t = layer(t)
        f = t.transpose(1, 2).reshape(B, D, Y, X)
        p = F.relu(self.proj(torch.cat([h, f], dim=1)))
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                z = p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                z = torch.cat([q[:, :, oy:oy + Y, ox:ox + X] for oy in range(dy) for ox in range(dx)], dim=1)
            res.append(out(F.relu(mix(z))))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
