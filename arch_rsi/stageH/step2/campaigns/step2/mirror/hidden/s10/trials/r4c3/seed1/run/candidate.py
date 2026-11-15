import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1}

LH = 32
CW = 32
P = 16
HID = 24
GD = 64
NH = 4


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        nin = 4 + 2 + 3 + Z
        self.lin = nn.Conv2d(nin, LH, 1)
        self.l2 = nn.Conv2d(3 * LH, LH, 1)
        self.cin = nn.Conv2d(2 * LH, CW, 1)
        self.c1 = nn.Conv2d(CW, CW, 3, padding=1)
        self.c2 = nn.Conv2d(CW, CW, 3, padding=1)
        self.ln1 = nn.LayerNorm(CW)
        self.qkv = nn.Linear(CW, 3 * CW)
        self.ao = nn.Linear(CW, CW)
        self.ln2 = nn.LayerNorm(CW)
        self.ff = nn.Sequential(nn.Linear(CW, 2 * CW), nn.ReLU(), nn.Linear(2 * CW, CW))
        self.proj = nn.Conv2d(LH, P, 1)
        self.cproj = nn.Conv2d(CW, P, 1)
        self.glob = nn.Sequential(nn.Linear(2 * LH + 2 * CW, GD), nn.ReLU())
        self.gin = nn.Linear(GD, HID * len(self.fps))
        self.mix = nn.ModuleList([nn.Conv2d(2 * P, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.done_head = nn.Linear(GD, 1)
        self.value_head = nn.Linear(GD, 1)

    def attend(self, cf):
        B, Cc, Y, X = cf.shape
        N = Y * X
        dh = Cc // NH
        t = cf.flatten(2).transpose(1, 2)
        q, k, v = self.qkv(self.ln1(t)).chunk(3, dim=-1)
        q = q.reshape(B, N, NH, dh).transpose(1, 2)
        k = k.reshape(B, N, NH, dh).transpose(1, 2)
        v = v.reshape(B, N, NH, dh).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dh), dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, N, Cc)
        t = t + self.ao(o)
        t = t + self.ff(self.ln2(t))
        return t.transpose(1, 2).reshape(B, Cc, Y, X)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        per = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).permute(0, 2, 1, 3, 4)
        below = F.pad(per[:, :-1, :2], (0, 0, 0, 0, 0, 0, 1, 0))
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        pos = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        inp = torch.cat([per, below, gl, pos], dim=2).reshape(B * Z, -1, Y, X)
        h = F.relu(self.lin(inp)).view(B, Z, LH, Y, X)
        up = torch.cummax(h, dim=1)[0]
        dn = torch.flip(torch.cummax(torch.flip(h, [1]), dim=1)[0], [1])
        up_ex = F.pad(up[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))
        dn_ex = F.pad(dn[:, 1:], (0, 0, 0, 0, 0, 0, 0, 1))
        h2 = F.relu(self.l2(torch.cat([h, up_ex, dn_ex], dim=2).reshape(B * Z, 3 * LH, Y, X)))
        h2 = h2 + h.reshape(B * Z, LH, Y, X)
        h2v = h2.view(B, Z, LH, Y, X)
        col = torch.cat([h2v.amax(1), h2v.mean(1)], dim=1)
        c = F.relu(self.cin(col))
        c = F.relu(self.c1(c))
        cf = F.relu(c + self.c2(c))
        cf = self.attend(cf)
        g = self.glob(torch.cat([h2v.mean((1, 3, 4)), h2v.amax((1, 3, 4)), cf.mean((2, 3)), cf.amax((2, 3))], dim=1))
        gb = self.gin(g)
        cp = self.cproj(cf).unsqueeze(1).expand(B, Z, P, Y, X).reshape(B * Z, P, Y, X)
        p = self.proj(h2) + cp
        res = []
        for i, ((dy, dx), mix, out) in enumerate(zip(self.fps, self.mix, self.outs)):
            if dy * dx == 1:
                mx, mn = p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            gi = gb[:, i * HID:(i + 1) * HID].reshape(B, 1, HID, 1, 1)
            hh = F.relu(mix(torch.cat([mx, mn], dim=1)).view(B, Z, HID, Y, X) + gi)
            s = out(hh.reshape(B * Z, HID, Y, X)).view(B, Z, Y, X)
            res.append(torch.logsumexp(s, dim=1).flatten(1))
        place = torch.cat(res, dim=1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
