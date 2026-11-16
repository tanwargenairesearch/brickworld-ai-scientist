import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.001}

STEM = 64
D = 48
HID = 32
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        self.nplace = int(spec.n_place)
        cin = int(spec.in_channels) + self.nt + 2
        self.inp = nn.Sequential(nn.Linear(cin, STEM), nn.ReLU(), nn.Linear(STEM, D))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(D, 4, 2 * D, dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(2)])
        self.norm = nn.LayerNorm(D)
        self.glob = nn.Sequential(nn.Linear(2 * D, G), nn.ReLU())
        self.hloc = nn.Conv2d(D, HID, 1)
        self.hglob = nn.Linear(G, HID)
        self.hout = nn.Conv2d(HID, self.nt, 3, padding=1)
        nn.init.normal_(self.hout.weight, std=0.01)
        nn.init.zeros_(self.hout.bias)
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        m = mask[:, :self.nplace].to(obs.dtype).reshape(B, self.nt, Y, X)
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        z = torch.cat([obs, m, yy, xx], dim=1).flatten(2).transpose(1, 2)
        t = self.inp(z)
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        f = t.transpose(1, 2).reshape(B, D, Y, X)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        h = F.relu(self.hloc(f) + self.hglob(g)[:, :, None, None])
        place = self.hout(h).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
