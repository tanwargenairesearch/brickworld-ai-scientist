import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 40
HEADS = 4
RECIPE = {"n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        W = WIDTH
        self.stem = nn.Conv2d(spec.in_channels + 2, W, 3, padding=1)
        self.conv1 = nn.Conv2d(W, W, 3, padding=1)
        self.ln = nn.LayerNorm(W)
        self.attn = nn.MultiheadAttention(W, HEADS, batch_first=True)
        self.conv2 = nn.Conv2d(W, W, 3, padding=1)
        self.head_hid = nn.Conv2d(W, 32, 1)
        self.place_head = nn.Conv2d(32, spec.n_types, 1)
        nn.init.normal_(self.place_head.weight, std=0.01)
        nn.init.zeros_(self.place_head.bias)
        self.trunk = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        x = torch.cat([obs, ys, xs], dim=1)
        h = F.relu(self.stem(x))
        h = h + F.relu(self.conv1(h))
        tok = h.flatten(2).transpose(1, 2)
        n = self.ln(tok)
        a, _ = self.attn(n, n, n, need_weights=False)
        h = h + a.transpose(1, 2).reshape(B, WIDTH, Y, X)
        h = F.relu(h + F.relu(self.conv2(h)))
        place = self.place_head(F.relu(self.head_hid(h))).flatten(1)
        t = self.trunk(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(t)], dim=1)
        return logits, self.value_head(t).squeeze(1)


def build_policy(spec):
    return Net(spec)
