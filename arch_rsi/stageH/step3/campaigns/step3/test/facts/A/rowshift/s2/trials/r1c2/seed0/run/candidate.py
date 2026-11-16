import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "ppo_epochs": 2, "gamma": 0.99, "gae_lambda": 0.9, "ent_coef": 0.005, "n_minibatches": 32}

VF = 32
CF = 32
SEL = 16
HID = 32
GH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.X = int(spec.X)
        self.Y = int(spec.Y)
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        Z = self.Z
        self.vox = nn.Sequential(nn.Linear(4 * 3 + 1 + 3, VF), nn.ReLU(), nn.Linear(VF, VF), nn.ReLU())
        self.cell = nn.Sequential(nn.Conv2d(spec.in_channels, CF, 1), nn.ReLU(),
                                  nn.Conv2d(CF, CF, 3, padding=1), nn.ReLU())
        self.sel_in = nn.Conv2d(CF, SEL, 1)
        self.sel = nn.ModuleList([nn.Sequential(nn.Conv2d(3 * SEL, SEL, 1), nn.ReLU(), nn.Conv2d(SEL, Z, 1))
                                  for _ in self.fps])
        self.mix = nn.ModuleList([nn.Sequential(nn.Conv2d(3 * VF + 3 * SEL, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
                                  for _ in self.fps])
        for m in self.mix:
            nn.init.zeros_(m[2].weight)
            nn.init.zeros_(m[2].bias)
        self.glob = nn.Sequential(nn.Linear(2 * CF + 2 * VF, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        nn.init.zeros_(self.done_head.weight)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Z, Y, X = self.Z, self.Y, self.X
        vox = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        vp = F.pad(vox, (0, 0, 0, 0, 1, 1))
        below = vp[:, :, 0:Z]
        above = vp[:, :, 2:Z + 2]
        zpos = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        hx = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        vin = torch.cat([below, vox, above, zpos, hx], dim=1)
        v = self.vox(vin.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

        c = self.cell(obs)
        p = self.sel_in(c)
        outs = []
        for t, (dy, dx) in enumerate(self.fps):
            if dy * dx == 1:
                sp = torch.cat([p, p, p], dim=1)
                vpad = v
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
                sp = torch.cat([mx, mn, av], dim=1)
                vpad = F.pad(v, (0, dx - 1, 0, dy - 1))
            w = torch.softmax(self.sel[t](sp), dim=1).unsqueeze(1)
            gs = []
            for oy in range(dy):
                for ox in range(dx):
                    vs = vpad[:, :, :, oy:oy + Y, ox:ox + X]
                    gs.append((vs * w).sum(2))
            gst = torch.stack(gs, 0)
            feat = torch.cat([gst.amax(0), gst.amin(0), gst.mean(0), sp], dim=1)
            outs.append(self.mix[t](feat))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([c.mean((2, 3)), c.amax((2, 3)), v.mean((2, 3, 4)), v.amax((2, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
