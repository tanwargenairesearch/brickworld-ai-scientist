import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_minibatches": 16, "lr": 0.001, "gamma": 0.9, "gae_lambda": 0.8, "ent_coef": 0.03}

H1 = 48
H2 = 32
PROJ = 24
HID = 16
G = 64
DONE_OFFSET = 2.0


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        Z = self.Z
        cin = 8 * Z + 3 + Z
        self.stem = nn.Sequential(
            nn.Conv2d(cin, H1, 1), nn.ReLU(),
            nn.Conv2d(H1, H2, 1), nn.ReLU())
        self.proj = nn.Conv2d(H2, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * H2 + Z * H2, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        nn.init.zeros_(self.done_head.weight)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        vox = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        pad = vox.new_zeros(B, 4, Z, Y, X)
        padded = torch.cat([pad, vox, pad], 2)
        rel = torch.stack([padded[:, :, z + 1:z + 1 + 2 * Z] for z in range(Z)], 1)
        rel = rel.reshape(B * Z, 8 * Z, Y, X)
        rest = obs[:, 4 * Z:].unsqueeze(1).expand(B, Z, 3, Y, X).reshape(B * Z, 3, Y, X)
        zoh = torch.eye(Z, device=obs.device, dtype=obs.dtype).reshape(1, Z, Z, 1, 1)
        zoh = zoh.expand(B, Z, Z, Y, X).reshape(B * Z, Z, Y, X)
        x = torch.cat([rel, rest, zoh], 1)
        f = self.stem(x)
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        s = torch.cat(res, 1)
        T = s.shape[1]
        s = s.reshape(B, Z, T, Y, X)
        place = torch.logsumexp(s, 1).flatten(1)
        fg = f.reshape(B, Z, -1, Y, X)
        g_in = torch.cat([fg.mean((1, 3, 4)), fg.amax((1, 3, 4)), fg.amax((3, 4)).flatten(1)], 1)
        g = self.glob(g_in)
        done = self.done_head(g) - DONE_OFFSET
        logits = torch.cat([place, done], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
