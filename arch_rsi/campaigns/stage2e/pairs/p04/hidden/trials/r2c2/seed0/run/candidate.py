import torch
import torch.nn as nn
import torch.nn.functional as F

W = 32
P = 16
H = 16

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2}


def zshift(t):
    # t: [B, Z, C, Y, X] -> concat of layer below, same layer, layer above along C
    p = F.pad(t, (0, 0, 0, 0, 0, 0, 1, 1))
    return torch.cat([p[:, :-2], p[:, 1:-1], p[:, 2:]], dim=2)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        c_in = 12 + 3 + 2
        self.conv1 = nn.Conv2d(c_in, W, 3, padding=1)
        self.zmix = nn.Conv2d(3 * W, W, 1)
        self.ctx = nn.Linear(2 * W, W)
        self.conv2 = nn.Conv2d(W, W, 3, padding=1)
        self.proj = nn.Conv2d(W, P, 1)
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * P, H, 1), nn.ReLU(), nn.Conv2d(H, 1, 1))
            for _ in self.fps
        ])
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        lay = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 2, 1, 3, 4)
        loc = zshift(lay)
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zr = torch.arange(Z, device=obs.device, dtype=obs.dtype)
        zc = (zr / max(Z - 1, 1)).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        z0 = (zr == 0).to(obs.dtype).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([loc, gl, zc, z0], dim=2).reshape(B * Z, -1, Y, X)

        h = F.relu(self.conv1(x))
        h5 = h.reshape(B, Z, W, Y, X)
        ctx = self.ctx(torch.cat([h5.mean((1, 3, 4)), h5.amax((1, 3, 4))], dim=1))
        m = self.zmix(zshift(h5).reshape(B * Z, 3 * W, Y, X)).reshape(B, Z, W, Y, X)
        h5 = h5 + F.relu(m + ctx[:, None, :, None, None])
        h = h5.reshape(B * Z, W, Y, X)
        h = h + F.relu(self.conv2(h))

        g = F.relu(self.proj(h))
        outs = []
        for (dy, dx), mlp in zip(self.fps, self.mlps):
            if dy > 1 or dx > 1:
                p = F.pad(g, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(p, (dy, dx), stride=1)
                mn = -F.max_pool2d(-p, (dy, dx), stride=1)
            else:
                mx = g
                mn = g
            outs.append(mlp(torch.cat([g, mx, mn], dim=1)))
        per_layer = torch.cat(outs, dim=1)
        T = per_layer.shape[1]
        place = per_layer.reshape(B, Z, T, Y, X).logsumexp(1).flatten(1)

        f5 = h.reshape(B, Z, W, Y, X)
        gg = self.glob(torch.cat([f5.mean((1, 3, 4)), f5.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
