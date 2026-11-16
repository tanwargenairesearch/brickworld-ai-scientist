import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

# layout = (type_major, spatial_transposed, flip_y, flip_x)
LAYOUTS = [(True, False, False, False), (True, False, False, True),
           (True, False, True, False), (True, False, True, True),
           (True, True, False, False), (True, True, False, True),
           (True, True, True, False), (True, True, True, True),
           (False, False, False, False), (False, False, False, True),
           (False, False, True, False), (False, False, True, True),
           (False, True, False, False), (False, True, False, True),
           (False, True, True, False), (False, True, True, True)]
OFFSETS = [0]
OBS_VIEWS = ["id"]
HEAD_K = 3
STEM = 64
WIDTH = 40
HEADC = 32
GLOB = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.grid_ok = self.n_place == self.T * self.Y * self.X
        self.views = [v for v in OBS_VIEWS if v != "tr" or self.Y == self.X]
        if self.grid_ok:
            self.offsets = [o for o in OFFSETS if o + self.n_place <= self.n_actions]
        else:
            self.offsets = [0]
        self.n_maps = len(self.offsets) * len(LAYOUTS)
        if self.grid_ok:
            mask_ch = self.n_maps * self.T
        else:
            mask_ch = 8
            self.mask_embed = nn.Linear(self.n_actions, 8 * self.Y * self.X)
        in_ch = len(self.views) * self.C + mask_ch + 2
        self.stem = nn.Sequential(nn.Conv2d(in_ch, STEM, 1), nn.ReLU(),
                                  nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        if HEAD_K == 3:
            self.hpre = nn.Identity()
            self.hconv = nn.Conv2d(WIDTH, HEADC, 3, padding=1)
        else:
            self.hpre = nn.Conv2d(WIDTH, 16, 1)
            self.hconv = nn.Conv2d(16, HEADC, HEAD_K, padding=HEAD_K // 2)
        self.hskip = nn.Conv2d(WIDTH, HEADC, 1)
        if self.grid_ok:
            self.hout = nn.Conv2d(HEADC, self.n_maps * self.T, 1)
        else:
            self.hout = nn.Conv2d(HEADC, 8, 1)
            self.fb_out = nn.Linear(8 * self.Y * self.X, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, len(self.offsets))
        self.value_head = nn.Linear(GLOB, 1)
        self.scale = min(1.0, 2.0 / (self.n_maps ** 0.5))

    def obs_views(self, obs):
        B = obs.shape[0]
        out = []
        for v in self.views:
            if v == "id":
                out.append(obs)
            elif v == "tr":
                out.append(obs.transpose(2, 3))
            elif v == "fy":
                out.append(obs.flip(2))
            elif v == "fx":
                out.append(obs.flip(3))
            elif v == "chl":
                out.append(obs.reshape(B, self.Y, self.X, self.C).permute(0, 3, 1, 2))
        return out

    def to_grid(self, v, lay):
        tmaj, tr, fy, fx = lay
        B = v.shape[0]
        T, Y, X = self.T, self.Y, self.X
        if tmaj:
            g = v.reshape(B, T, X, Y).transpose(2, 3) if tr else v.reshape(B, T, Y, X)
        else:
            g = v.reshape(B, X, Y, T).permute(0, 3, 2, 1) if tr else v.reshape(B, Y, X, T).permute(0, 3, 1, 2)
        if fy:
            g = g.flip(2)
        if fx:
            g = g.flip(3)
        return g

    def from_grid(self, g, lay):
        tmaj, tr, fy, fx = lay
        B = g.shape[0]
        if fy:
            g = g.flip(2)
        if fx:
            g = g.flip(3)
        if tmaj:
            if tr:
                g = g.transpose(2, 3)
        else:
            g = g.permute(0, 3, 2, 1) if tr else g.permute(0, 2, 3, 1)
        return g.reshape(B, -1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask.float()
        parts = self.obs_views(obs)
        if self.grid_ok:
            for o in self.offsets:
                v = m[:, o:o + self.n_place]
                for lay in LAYOUTS:
                    parts.append(self.to_grid(v, lay))
        else:
            parts.append(self.mask_embed(m).view(B, 8, self.Y, self.X))
        ys = torch.linspace(-1.0, 1.0, self.Y, device=obs.device, dtype=obs.dtype).view(1, 1, self.Y, 1).expand(B, 1, self.Y, self.X)
        xs = torch.linspace(-1.0, 1.0, self.X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, self.X).expand(B, 1, self.Y, self.X)
        parts.append(ys)
        parts.append(xs)
        h = self.stem(torch.cat(parts, 1))
        h = F.relu(h + self.r1(h))
        f = F.relu(h + self.r2(h))
        hh = F.relu(self.hconv(self.hpre(f)) + self.hskip(f))
        maps = self.hout(hh)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        done = self.done_head(g)
        if self.grid_ok:
            logits = obs.new_zeros(B, self.n_actions)
            k = 0
            for i, o in enumerate(self.offsets):
                acc = 0
                for lay in LAYOUTS:
                    acc = acc + self.from_grid(maps[:, k * self.T:(k + 1) * self.T], lay)
                    k += 1
                logits = logits + F.pad(acc * self.scale, (o, self.n_actions - self.n_place - o))
                pos = self.n_place if o == 0 else 0
                logits = logits + F.pad(done[:, i:i + 1], (pos, self.n_actions - 1 - pos))
        else:
            place = self.fb_out(maps.flatten(1))
            logits = torch.cat([place, done[:, :1]], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
