import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 32, "n_minibatches": 32, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.995, "gae_lambda": 0.95, "clip_eps": 0.3}

CTX = 16
K = 16
L1 = 32
HID = 32
G = 64
RG = math.sqrt(2.0)


def init_(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def norm(c):
    return nn.GroupNorm(8 if c % 8 == 0 else 1, c)


def fp_pool(p, dy, dx):
    if dy * dx == 1:
        return [p, p, p]
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return [mx, mn, av]


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        nt = len(self.fps)
        self.c_stem = init_(nn.Conv2d(cin, 48, 1), RG)
        self.c_conv = init_(nn.Conv2d(48, 32, 3, padding=1), RG)
        self.c_norm = norm(32)
        self.c_out = init_(nn.Conv2d(32, CTX, 1), 1.0)
        lin = 12 + 3 + 1 + CTX
        self.l1 = init_(nn.Conv2d(lin, L1, 1), RG)
        self.ln1 = norm(L1)
        self.l2 = init_(nn.Conv2d(L1, K, 1), 1.0)
        self.mix = nn.ModuleList([init_(nn.Conv2d(3 * K, HID, 1), RG) for _ in self.fps])
        self.outs = nn.ModuleList([init_(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        gin = 64 + 2 * K
        self.gnorm = nn.LayerNorm(gin)
        self.glob = init_(nn.Linear(gin, G), RG)
        self.type_head = init_(nn.Linear(G, nt), 0.01)
        self.done_head = init_(nn.Linear(G, 1), 0.01)
        self.value_head = init_(nn.Linear(G, 1), 1.0)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        c = F.relu(self.c_stem(obs))
        c = F.relu(self.c_norm(self.c_conv(c)))
        ctx = self.c_out(c)
        per = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        below = F.pad(per, (0, 0, 0, 0, 1, 0))[:, :, :Z]
        above = F.pad(per, (0, 0, 0, 0, 0, 1))[:, :, 1:]
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        pos = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        cx = ctx.unsqueeze(2).expand(B, CTX, Z, Y, X)
        a = torch.cat([per, below, above, gl, pos, cx], dim=1)
        a = a.permute(0, 2, 1, 3, 4).reshape(B * Z, -1, Y, X)
        k = F.relu(self.ln1(self.l1(a)))
        k = self.l2(k)
        kk = k.view(B, Z, K, Y, X)
        g = torch.cat([c.mean((2, 3)), c.amax((2, 3)), kk.amax(dim=(1, 3, 4)), kk.mean(dim=(1, 3, 4))], dim=1)
        g = F.relu(self.glob(self.gnorm(g)))
        tb = self.type_head(g)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            h = F.relu(self.mix[i](torch.cat(fp_pool(k, dy, dx), dim=1)))
            s = self.outs[i](h).view(B, Z, Y, X)
            res.append(torch.logsumexp(s, dim=1, keepdim=True) + tb[:, i].view(-1, 1, 1, 1))
        place = torch.cat(res, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
