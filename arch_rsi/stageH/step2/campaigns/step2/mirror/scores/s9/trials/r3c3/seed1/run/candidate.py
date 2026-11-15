import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 32, "n_minibatches": 16, "lr": 0.0007, "gamma": 0.9, "gae_lambda": 0.9, "ent_coef": 0.002}

D = 48
HEADS = 4
LAYERS = 2
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        cin = spec.in_channels + self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, D, 1), nn.ReLU(),
            nn.Conv2d(D, D, 3, padding=1))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(D, HEADS, 2 * D, dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(LAYERS)])
        self.norm = nn.LayerNorm(D)
        self.h1 = nn.Conv2d(D, HID, 3, padding=1)
        self.h2 = nn.Conv2d(HID, self.T, 1)
        nn.init.normal_(self.h2.weight, std=0.01)
        nn.init.zeros_(self.h2.bias)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        m = mask[:, :self.T * Y * X].to(obs.dtype).reshape(B, self.T, Y, X)
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).reshape(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).reshape(1, 1, 1, X).expand(B, 1, Y, X)
        x = self.stem(torch.cat([obs, m, ys, xs], dim=1))
        t = x.flatten(2).transpose(1, 2)
        for layer in self.layers:
            t = layer(t)
        t = self.norm(t)
        f = t.transpose(1, 2).reshape(B, D, Y, X)
        place = self.h2(F.relu(self.h1(f))).flatten(1)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
