import torch
import torch.nn as nn
import torch.nn.functional as F

W3 = 16
HID = 16
GW = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.body = nn.Sequential(
            nn.Conv3d(8, W3, 3, padding=1), nn.ReLU(),
            nn.Conv3d(W3, W3, 3, padding=1), nn.ReLU())
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.fp_convs = nn.ModuleList([nn.Conv3d(W3, HID, (1, dy, dx)) for dy, dx in self.fps])
        self.fp_outs = nn.ModuleList([nn.Conv3d(HID, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W3, GW), nn.ReLU())
        self.done_head = nn.Linear(GW, 1)
        self.value_head = nn.Linear(GW, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        per = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        gch = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        zc = (torch.arange(Z, dtype=obs.dtype, device=obs.device) / max(Z - 1, 1)).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        x = torch.cat([per, gch, zc], dim=1)
        f = self.body(x)
        outs = []
        for i, (dy, dx) in enumerate(self.fps):
            fp = F.pad(f, (0, dx - 1, 0, dy - 1))
            h = F.relu(self.fp_convs[i](fp))
            o = self.fp_outs[i](h)
            score = o[:, 0]
            w = torch.softmax(o[:, 1], dim=1)
            outs.append((w * score).sum(1))
        place = torch.stack(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3, 4)), f.amax((2, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
