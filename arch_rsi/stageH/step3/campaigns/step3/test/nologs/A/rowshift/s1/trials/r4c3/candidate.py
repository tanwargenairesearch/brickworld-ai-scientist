import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.95, "gae_lambda": 0.9, "lr": 0.0007, "ent_coef": 0.005}

D = 48
FF = 96
HEADS = 4
LAYERS = 2
HID = 64


class Layer(nn.Module):
    def __init__(self, d, ff, heads):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, ff)
        self.f2 = nn.Linear(ff, d)

    def forward(self, x):
        B, N, d = x.shape
        qkv = self.qkv(self.n1(x)).reshape(B, N, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.o(a.transpose(1, 2).reshape(B, N, d))
        x = x + self.f2(F.relu(self.f1(self.n2(x))))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.kh = max(int(a) for a, b in spec.footprints)
        self.kw = max(int(b) for a, b in spec.footprints)
        cin = spec.in_channels + self.T
        self.embed = nn.Sequential(
            nn.Conv2d(cin, 64, 1), nn.ReLU(),
            nn.Conv2d(64, D, 1))
        self.pos = nn.Parameter(torch.zeros(1, D, self.Y, self.X))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList([Layer(D, FF, HEADS) for _ in range(LAYERS)])
        self.nf = nn.LayerNorm(D)
        self.foot = nn.Conv2d(D, 32, (self.kh, self.kw))
        self.out = nn.Conv2d(32, self.T, 1)
        self.g = nn.Sequential(nn.Linear(2 * D, HID), nn.ReLU())
        self.done = nn.Linear(HID, 1)
        self.value = nn.Linear(HID, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        m = mask[:, :self.n_place].reshape(B, self.T, Y, X).to(obs.dtype)
        x = torch.cat([obs, m], dim=1)
        h = self.embed(x) + self.pos
        t = h.flatten(2).transpose(1, 2)
        for layer in self.layers:
            t = layer(t)
        t = self.nf(t)
        h = t.transpose(1, 2).reshape(B, D, Y, X)
        q = F.pad(h, (0, self.kw - 1, 0, self.kh - 1))
        p = F.relu(self.foot(q))
        place = self.out(p).flatten(1)
        g = self.g(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([place, self.done(g)], dim=1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
