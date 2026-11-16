import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 2, "ent_coef": 0.02, "gamma": 0.9, "gae_lambda": 0.85}

STEM = 64
WIDTH = 40
HEADW = 32
NBLOCKS = 2
GBCAST = False


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.ny = int(spec.Y)
        self.nx = int(spec.X)
        self.nt = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        cin0 = int(spec.in_channels)
        cells = self.ny * self.nx
        self.grid_ok = self.n_place == self.nt * cells
        if self.grid_ok:
            n_mask = self.nt
        else:
            n_mask = 8
            self.mask_emb = nn.Linear(self.n_actions, 8 * cells)
            self.place_fc = nn.Linear(self.nt * cells, self.n_place)
        cin = cin0 + n_mask + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(NBLOCKS)])
        self.hk = nn.Conv2d(WIDTH, HEADW, 3, padding=1)
        self.hs = nn.Conv2d(WIDTH, HEADW, 1)
        self.out = nn.Conv2d(HEADW, self.nt, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        if GBCAST:
            self.gproj = nn.Linear(64, HEADW)
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        b = obs.shape[0]
        ny, nx, nt = self.ny, self.nx, self.nt
        m = mask.to(obs.dtype)
        feats = [obs]
        if self.grid_ok:
            feats.append(torch.flip(m[:, :self.n_place].reshape(b, nt, ny, nx), [2]))
        else:
            feats.append(self.mask_emb(m).reshape(b, 8, ny, nx))
        yy = torch.linspace(-1.0, 1.0, ny, device=obs.device, dtype=obs.dtype).view(1, 1, ny, 1).expand(b, 1, ny, nx)
        xx = torch.linspace(-1.0, 1.0, nx, device=obs.device, dtype=obs.dtype).view(1, 1, 1, nx).expand(b, 1, ny, nx)
        feats.append(yy)
        feats.append(xx)
        f = self.stem(torch.cat(feats, 1))
        for blk in self.blocks:
            f = F.relu(f + blk(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        hpre = self.hk(f) + self.hs(f)
        if GBCAST:
            hpre = hpre + self.gproj(g)[:, :, None, None]
        h = F.relu(hpre)
        value = self.value_head(g).squeeze(1)
        if self.grid_ok:
            place = torch.flip(self.out(h), [2]).reshape(b, -1)
        else:
            place = self.place_fc(self.out(h).reshape(b, -1))
        logits = torch.cat([place, self.done_head(g)], 1)
        if logits.shape[1] < self.n_actions:
            logits = F.pad(logits, (0, self.n_actions - logits.shape[1]))
        return logits, value


def build_policy(spec):
    return Net(spec)
