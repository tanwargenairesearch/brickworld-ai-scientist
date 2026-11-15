import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "ppo_epochs": 2, "n_minibatches": 16, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005}

H1 = 48
H2 = 32
CTX = 32
MIX = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        c_in = spec.in_channels
        self.layer_in = nn.Conv2d(c_in, self.Z * H1, 1)
        self.layer_h = nn.Conv2d(H1, H2, 1)
        self.ctx1 = nn.Conv2d(c_in, CTX, 1)
        self.ctx2 = nn.Conv2d(CTX, CTX, 3, padding=1)
        self.ctx_to_layer = nn.Conv2d(CTX, H2, 1)
        self.mix = nn.ModuleList([nn.Conv2d(2 * H2, MIX, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(MIX, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * CTX + 2 * H2, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        Z = self.Z
        c = F.relu(self.ctx1(obs))
        c = F.relu(c + self.ctx2(c))
        h = F.relu(self.layer_in(obs))
        h = h.reshape(B * Z, H1, Y, X)
        h = self.layer_h(h).reshape(B, Z, H2, Y, X)
        h = h + self.ctx_to_layer(c).unsqueeze(1)
        h5 = F.relu(h)
        h = h5.reshape(B * Z, H2, Y, X)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn = h, h
            else:
                q = F.pad(h, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            res.append(out(F.relu(mix(torch.cat([mx, mn], dim=1)))))
        s = torch.cat(res, dim=1).reshape(B, Z, len(self.fps), Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        g_in = torch.cat([c.mean((2, 3)), c.amax((2, 3)),
                          h5.mean((1, 3, 4)), h5.amax((1, 3, 4))], dim=1)
        g = self.glob(g_in)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
