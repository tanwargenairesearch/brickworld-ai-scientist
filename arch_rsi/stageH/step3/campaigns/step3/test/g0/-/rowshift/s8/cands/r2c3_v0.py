import torch
import torch.nn as nn
import torch.nn.functional as F

H1 = 48
H2 = 32
PROJ = 16
HID = 16
GH = 64

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        Z = self.Z
        cin = 4 * (2 * Z + 1) + 3 + Z + self.T
        self.inp = nn.Conv2d(cin, H1, 1)
        self.l2 = nn.Conv2d(H1, H2, 1)
        self.ctx = nn.Conv2d(H2, H2, 3, padding=1)
        self.proj = nn.Conv2d(H2, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * H2, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)
        self.register_buffer('zeye', torch.eye(Z))

    def layer_view(self, obs, mask):
        B, Y, X = obs.shape[0], obs.shape[2], obs.shape[3]
        Z = self.Z
        T = self.T
        vol = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        fill = vol[:, :2]
        diff = vol[:, 2:]
        fill = F.pad(fill, (0, 0, 0, 0, Z, 0), value=1.0)
        fill = F.pad(fill, (0, 0, 0, 0, 0, Z), value=0.0)
        diff = F.pad(diff, (0, 0, 0, 0, Z, Z), value=0.0)
        vp = torch.cat([fill, diff], dim=1)
        views = torch.stack([vp[:, :, z:z + 2 * Z + 1] for z in range(Z)], dim=1)
        views = views.reshape(B * Z, 4 * (2 * Z + 1), Y, X)
        g = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X).reshape(B * Z, 3, Y, X)
        lid = self.zeye.to(obs.dtype).reshape(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X).reshape(B * Z, Z, Y, X)
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, 1, T, Y, X).expand(B, Z, T, Y, X).reshape(B * Z, T, Y, X)
        return torch.cat([views, g, lid, m], dim=1)

    def forward(self, obs, mask):
        B, Y, X = obs.shape[0], obs.shape[2], obs.shape[3]
        Z = self.Z
        h = self.layer_view(obs, mask)
        h = F.relu(self.inp(h))
        h = F.relu(self.l2(h))
        h = F.relu(h + self.ctx(h))
        p = self.proj(h)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            u = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(u))
        s = torch.cat(res, dim=1)
        T = s.shape[1]
        s = s.reshape(B, Z, T, Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        hz = h.reshape(B, Z, H2, Y, X)
        gf = torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1)
        g = self.glob(gf)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
