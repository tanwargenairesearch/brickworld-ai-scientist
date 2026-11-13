"""Per-layer weight-shared residual 2D conv seeing layers z-1, z, z+1; footprint logits per layer combined by logsumexp over z."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 20


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        in_c = 12 + 3 + spec.Z
        self.c1 = nn.Conv2d(in_c, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c3 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.fps = [(int(f[0]), int(f[1])) for f in spec.footprints]
        self.heads = nn.ModuleList([nn.Conv2d(WIDTH, 1, (dy, dx)) for dy, dx in self.fps])
        for h in self.heads:
            nn.init.normal_(h.weight, std=0.01)
            nn.init.zeros_(h.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        cur = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 2, 1, 3, 4)
        padded = F.pad(cur, (0, 0, 0, 0, 0, 0, 1, 1))
        below = padded[:, 0:Z]
        above = padded[:, 2:Z + 2]
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zoh = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        x = torch.cat([below, cur, above, gl, zoh], dim=2).reshape(B * Z, 12 + 3 + Z, Y, X)
        h = F.relu(self.c1(x))
        r = F.relu(self.c2(h))
        h = F.relu(h + self.c3(r))
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            p = F.pad(h, (0, dx - 1, 0, dy - 1))
            outs.append(head(p))
        per_layer = torch.cat(outs, dim=1).reshape(B, Z, len(self.fps), Y, X)
        place = torch.logsumexp(per_layer, dim=1).flatten(1)
        hz = h.reshape(B, Z, -1, Y, X)
        pooled = torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1)
        g = self.glob(pooled)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
