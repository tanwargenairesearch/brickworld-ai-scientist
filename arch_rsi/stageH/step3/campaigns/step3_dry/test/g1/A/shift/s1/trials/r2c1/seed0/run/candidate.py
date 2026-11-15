import torch
import torch.nn as nn
import torch.nn.functional as F

W = 48


def layout_tables(T, Y, X):
    n_place = T * Y * X
    n_act = n_place + 1
    fwd = []
    inv = []
    for shift in (0, 1):
        for tmajor in (True, False):
            for rowmajor in (True, False):
                for fy in (False, True):
                    for fx in (False, True):
                        idx = [n_place] * n_act
                        back = [0] * n_place
                        for a in range(n_act):
                            if shift == 0:
                                if a == n_place:
                                    continue
                                q = a
                            else:
                                if a == 0:
                                    continue
                                q = a - 1
                            if tmajor:
                                t = q // (Y * X)
                                p = q % (Y * X)
                            else:
                                t = q % T
                                p = q // T
                            if rowmajor:
                                y = p // X
                                x = p % X
                            else:
                                x = p // Y
                                y = p % Y
                            if fy:
                                y = Y - 1 - y
                            if fx:
                                x = X - 1 - x
                            j = t * Y * X + y * X + x
                            idx[a] = j
                            back[j] = a
                        fwd.append(idx)
                        inv.append(back)
    return torch.tensor(fwd, dtype=torch.long), torch.tensor(inv, dtype=torch.long)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = spec.n_types
        self.Y = spec.Y
        self.X = spec.X
        self.n_place = spec.n_place
        fwd, inv = layout_tables(self.T, self.Y, self.X)
        self.K = fwd.shape[0]
        self.register_buffer("fwd", fwd)
        self.register_buffer("inv", inv)
        self.stem = nn.Conv2d(spec.in_channels + self.K * self.T, W, 1)
        self.convs = nn.ModuleList([nn.Conv2d(W, W, 3, padding=1) for _ in range(3)])
        self.out = nn.Conv2d(W, self.K * self.T, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done = nn.Linear(64, self.K)
        nn.init.zeros_(self.done.weight)
        nn.init.zeros_(self.done.bias)
        self.value = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask.float()[:, self.inv].reshape(B, self.K * self.T, self.Y, self.X)
        h = F.relu(self.stem(torch.cat([obs, m], dim=1)))
        for c in self.convs:
            h = F.relu(h + c(h))
        o = self.out(h).reshape(B, self.K, self.n_place)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        e = torch.cat([o, self.done(g).unsqueeze(2)], dim=2)
        idx = self.fwd.unsqueeze(0).expand(B, -1, -1)
        logits = torch.gather(e, 2, idx).sum(1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
