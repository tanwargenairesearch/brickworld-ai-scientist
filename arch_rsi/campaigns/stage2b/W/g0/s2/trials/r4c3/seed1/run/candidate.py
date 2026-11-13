import torch
import torch.nn as nn
import torch.nn.functional as F

COL = 64
WIDTH = 48
P = 32
H = 32
RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "gamma": 0.99, "gae_lambda": 0.95}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.my = max(fp[0] for fp in self.fps) - 1
        self.mx = max(fp[1] for fp in self.fps) - 1
        self.col = nn.Sequential(
            nn.Conv2d(spec.in_channels, COL, 1), nn.ReLU(),
            nn.Conv2d(COL, COL, 1), nn.ReLU())
        self.body = nn.Sequential(
            nn.Conv2d(COL, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.place_head = nn.Conv2d(WIDTH + COL, spec.n_types, 1)
        self.proj = nn.Sequential(nn.Conv2d(WIDTH + COL, P, 1), nn.ReLU())
        self.type_mlp = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2 * P, H, 1), nn.ReLU(), nn.Conv2d(H, 1, 1))
            for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * (WIDTH + COL), 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        c = self.col(obs)
        f = torch.cat([self.body(c), c], dim=1)
        p = self.proj(f)
        pp = F.pad(p, (0, self.mx, 0, self.my))
        outs = []
        for t, (dy, dx) in enumerate(self.fps):
            sh = [pp[:, :, i:i + Y, j:j + X] for i in range(dy) for j in range(dx)]
            st = torch.stack(sh, 0)
            a = torch.cat([st.amax(0), st.mean(0)], dim=1)
            outs.append(self.type_mlp[t](a))
        place = (self.place_head(f) + torch.cat(outs, dim=1)).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
