import torch
import torch.nn as nn
import torch.nn.functional as F

C = 64
L = 32
H = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, C, 1), nn.ReLU(),
            nn.Conv2d(C, C, 1), nn.ReLU())
        self.col_proj = nn.Conv2d(C, L, 1)
        self.loc_in = nn.Conv2d(12 + spec.Z, L, 1)
        self.loc2 = nn.Conv2d(L, L, 1)
        self.register_buffer("zpos", torch.eye(spec.Z).view(1, spec.Z, spec.Z, 1, 1))
        heads = []
        for dy, dx in self.fps:
            out = nn.Conv2d(H, 1, 1)
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
            heads.append(nn.Sequential(nn.Conv2d(L, H, (dy, dx)), nn.ReLU(), out))
        self.heads = nn.ModuleList(heads)
        self.glob = nn.Sequential(nn.Linear(2 * C + 2 * L, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        nn.init.zeros_(self.done_head.weight)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        layers = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).transpose(1, 2)
        below = F.pad(layers, (0, 0, 0, 0, 0, 0, 1, 0))[:, :Z]
        above = F.pad(layers, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:]
        zp = self.zpos.expand(B, Z, Z, Y, X)
        x = torch.cat([below, layers, above, zp], dim=2).reshape(B * Z, 12 + Z, Y, X)
        col = self.stem(obs)
        cp = self.col_proj(col)
        h = self.loc_in(x).view(B, Z, L, Y, X) + cp.unsqueeze(1)
        h = F.relu(h).reshape(B * Z, L, Y, X)
        h = F.relu(self.loc2(h))
        outs = []
        for fp, head in zip(self.fps, self.heads):
            dy, dx = fp
            p = F.pad(h, (0, dx - 1, 0, dy - 1))
            s = head(p).view(B, Z, Y * X)
            outs.append(torch.logsumexp(s, dim=1))
        place = torch.cat(outs, dim=1)
        hz = h.view(B, Z, L, Y, X)
        g = torch.cat([col.mean((2, 3)), col.amax((2, 3)),
                       hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1)
        g = self.glob(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
