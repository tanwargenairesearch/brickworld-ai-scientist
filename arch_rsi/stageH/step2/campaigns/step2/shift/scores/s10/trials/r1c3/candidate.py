import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 1, "ent_coef": 0.003}

S1 = 32
D = 32
HID = 16
CTX = 32
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        Z = self.Z
        cin = 4 * (2 * Z - 1) + 4
        self.stem = nn.Sequential(
            nn.Conv2d(cin, S1, 1), nn.ReLU(),
            nn.Conv2d(S1, D, 1), nn.ReLU())
        self.ctx = nn.Conv2d(D, CTX, 3, padding=1)
        self.ctx_back = nn.Conv2d(CTX, D, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * D, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * D + 2 * CTX, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def layer_input(self, obs):
        Z = self.Z
        B, C, Y, X = obs.shape
        grp = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], dim=1)
        grp = F.pad(grp, (0, 0, 0, 0, Z - 1, Z - 1))
        wins = torch.stack([grp[:, :, z:z + 2 * Z - 1] for z in range(Z)], dim=1)
        wins = wins.reshape(B, Z, 4 * (2 * Z - 1), Y, X)
        rest = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zpos = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        inp = torch.cat([wins, rest, zpos], dim=2)
        return inp.reshape(B * Z, -1, Y, X)

    def forward(self, obs, mask):
        Z = self.Z
        B, C, Y, X = obs.shape
        h = self.stem(self.layer_input(obs))
        h5 = h.view(B, Z, D, Y, X)
        c = F.relu(self.ctx(h5.amax(1)))
        back = self.ctx_back(c)
        f5 = F.relu(h5 + back.unsqueeze(1))
        f = f5.reshape(B * Z, D, Y, X)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = f, f, f
            else:
                q = F.pad(f, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            res.append(out(F.relu(mix(torch.cat([mx, mn, av], dim=1)))))
        s = torch.cat(res, dim=1).view(B, Z, len(self.fps), Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        g = self.glob(torch.cat([f5.mean((1, 3, 4)), f5.amax((1, 3, 4)), c.mean((2, 3)), c.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
