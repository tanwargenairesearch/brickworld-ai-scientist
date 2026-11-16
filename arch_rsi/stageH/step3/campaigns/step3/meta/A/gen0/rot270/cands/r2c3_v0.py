import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0007, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95}

D = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.X = int(spec.X)
        self.Y = int(spec.Y)
        self.nt = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.embed = nn.Linear(spec.in_channels + self.nt, D)
        self.pos = nn.Parameter(torch.randn(1, self.Y * self.X, D) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(D, 4, 128, dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(2)])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Sequential(nn.Conv2d(D, 32, 1), nn.ReLU(), nn.Conv2d(32, self.nt, 3, padding=1))
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.nt, Y, X)
        x = torch.cat([obs, m], dim=1).flatten(2).transpose(1, 2)
        t = self.embed(x) + self.pos
        for layer in self.layers:
            t = layer(t)
        t = self.norm(t)
        f = t.transpose(1, 2).reshape(B, D, Y, X)
        place = self.head(f).flatten(1)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
