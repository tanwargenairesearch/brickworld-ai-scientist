import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.01, "lr": 0.001}

W = 40
D = 24
NH = 6
HID = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.N = self.Y * self.X
        self.nt = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + self.nt, 64, 1), nn.ReLU(),
            nn.Conv2d(64, W, 1), nn.ReLU())
        self.blocks = nn.ModuleList([nn.Conv2d(W, W, 3, padding=1) for _ in range(2)])
        self.feat = nn.Conv2d(W, D, 1)
        self.gather = nn.Parameter(torch.randn(NH, self.N, self.N))
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.amlp1 = nn.Linear(NH * D + 64 + self.nt, HID)
        self.amlp2 = nn.Linear(HID, self.nt)
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        mflat = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.nt, self.N)
        m = mflat.reshape(B, self.nt, self.Y, self.X)
        f = self.stem(torch.cat([obs, m], dim=1))
        for blk in self.blocks:
            f = F.relu(f + blk(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        v = self.feat(f).flatten(2).transpose(1, 2)
        A = torch.softmax(self.gather, dim=-1)
        u = torch.einsum('han,bnd->bahd', A, v).reshape(B, self.N, NH * D)
        u = torch.cat([u, g.unsqueeze(1).expand(B, self.N, 64), mflat.transpose(1, 2)], dim=2)
        z = self.amlp2(F.relu(self.amlp1(u)))
        place = z.transpose(1, 2).reshape(B, -1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
