import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 256, "rollout_len": 16, "n_minibatches": 32, "ppo_epochs": 1, "lr": 0.0007, "ent_coef": 0.003, "gamma": 0.93, "gae_lambda": 0.9}

D = 64
HID = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.P = int(spec.n_place)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cin = int(spec.in_channels) + self.T
        self.e1 = nn.Conv2d(cin, D, 1)
        self.e2 = nn.Conv2d(D, D, 1)
        self.pos = nn.Parameter(torch.zeros(1, self.Y * self.X, D))
        nn.init.normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(D, 4, 128, dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(2)])
        self.norm = nn.LayerNorm(D)
        self.fconv = nn.ModuleList([nn.Conv2d(D, HID, (dy, dx)) for dy, dx in self.fps])
        self.fout = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = self.Y
        X = self.X
        m = mask[:, :self.P].to(obs.dtype).reshape(B, self.T, Y, X)
        inp = torch.cat([obs, m], dim=1)
        s = F.relu(self.e1(inp))
        s = self.e2(s)
        t = s.flatten(2).transpose(1, 2) + self.pos
        for layer in self.layers:
            t = layer(t)
        t = self.norm(t)
        f = t.transpose(1, 2).reshape(B, D, Y, X)
        res = []
        for (dy, dx), cv, out in zip(self.fps, self.fconv, self.fout):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            res.append(out(F.relu(cv(q))))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
