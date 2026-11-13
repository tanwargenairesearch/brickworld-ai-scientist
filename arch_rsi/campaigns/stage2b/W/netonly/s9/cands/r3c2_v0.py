import torch
import torch.nn as nn
import torch.nn.functional as F

H = 48
CUM = 12
CTX = 24
P = 16
HID = 16


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        g = 2.0 ** 0.5
        self.Z = int(spec.Z)
        Z = self.Z
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.l1 = init_layer(nn.Linear(15 + Z, H), g)
        self.l2 = init_layer(nn.Linear(H, CUM), g)
        self.ctx = nn.Sequential(
            init_layer(nn.Conv2d(spec.in_channels, CTX, 3, padding=1), g), nn.ReLU(),
            init_layer(nn.Conv2d(CTX, CTX, 3, padding=1), g), nn.ReLU())
        self.l3 = init_layer(nn.Linear(H + 2 * CUM + CTX, P), g)
        heads = []
        for _ in self.footprints:
            heads.append(nn.Sequential(
                init_layer(nn.Conv2d(3 * P, HID, 1), g), nn.ReLU(),
                init_layer(nn.Conv2d(HID, 1, 1), 0.01)))
        self.type_heads = nn.ModuleList(heads)
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * P + 2 * CTX, 64), g), nn.ReLU())
        self.done_head = init_layer(nn.Linear(64, 1), 0.01)
        self.value_head = init_layer(nn.Linear(64, 1), 1.0)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 2, 1, 3, 4)
        below = F.pad(s, (0, 0, 0, 0, 0, 0, 1, 0))[:, :Z]
        above = F.pad(s, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:]
        g3 = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zoh = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        x = torch.cat([s, below, above, g3, zoh], dim=2).permute(0, 1, 3, 4, 2)
        u = torch.relu(self.l1(x))
        v = torch.relu(self.l2(u))
        cs = torch.cumsum(v, dim=1)
        below_c = (cs - v) / Z
        above_c = (cs[:, -1:] - cs) / Z
        c = self.ctx(obs)
        cz = c.permute(0, 2, 3, 1).unsqueeze(1).expand(B, Z, Y, X, CTX)
        h = torch.relu(self.l3(torch.cat([u, below_c, above_c, cz], dim=-1)))
        p = h.permute(0, 1, 4, 2, 3).reshape(B * Z, P, Y, X)
        outs = []
        for fp, head in zip(self.footprints, self.type_heads):
            dy, dx = fp
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(q, (dy, dx), stride=1)
            av = F.avg_pool2d(q, (dy, dx), stride=1)
            mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            outs.append(head(torch.cat([mx, av, mn], dim=1)))
        score = torch.cat(outs, dim=1)
        T = score.shape[1]
        score = score.reshape(B, Z, T, Y, X)
        place = torch.logsumexp(score, dim=1).flatten(1)
        pooled = torch.cat([h.mean((1, 2, 3)), h.amax((1, 2, 3)),
                            c.mean((2, 3)), c.amax((2, 3))], dim=1)
        g = self.glob(pooled)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
