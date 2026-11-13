"""The paper's focused search with each of its stopping rules (paper_rules.py), as a custom Metal kernel for Apple
GPUs, compiled with torch.mps.compile_shader.

One GPU thread runs one chain: it owns one row of the label buffer `y`, of the token buffer `tok` and of the flag
buffer `flag`, so no two threads write the same memory. A launch makes `nmoves` moves of every chain that has not
stopped. Every launch begins by recomputing each cell's flags and the board energy from the labels, so rounding in
the running energy cannot build up beyond one launch.

Per filled cell the kernel keeps two flags: bit 0 `hot` (the search's focus) and bit 1 `fail` (the stopping rule
does not hold at this cell). With L[a] the local energy of label a at the cell, own = L[its label] and
margin = (lowest L of another label) - own:

  rule 0 (tau)    hot: own > focus_at       stop: board energy <= thr                  thr = tau + 1e-3
  rule 1 (terms)  hot: own > focus_at       fail: a term of the cell is above its threshold (tables bad8, badB)
  rule 2 (local)  hot: margin < 0           fail: margin < thr                         thr = delta - 1e-3
  rule 3 (fixed)  hot: margin < 0           fail: hot

One move of a chain:
  1. stop if the rule holds (rule 0: the energy test; the others: no cell fails);
  2. pick the r-th hot cell, or the r-th filled cell when none is hot; r = hash(seed, chain, move, 0) mod the count;
  3. the new label: argmax over a of -L[a] / temp + Gumbel noise (temp 0: the lowest L[a], the first on a tie);
  4. recompute the flags of the cell and of its 8 neighbours.
The moves are those of `paper_rules.search(..., hashed=True)`, with the same counter-hash random numbers.

`found`: -1 running, -2 cancelled by the host (another chain of its design stopped), >= 0 the moves made at the stop.
The kernel works in float32 and paper_rules.py in float64, so the host judges every stop again in float64.
"""
import time

import numpy as np
import torch

import brick_cram as BC
import paper_rules as PR

RULE_ID = {"tau": 0, "terms": 1, "terms1": 1, "local": 2, "fixed": 3}

SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define NC 26

constant int DY[8] = {0, 1, 1, 1, 0, -1, -1, -1};
constant int DX[8] = {1, -1, 0, 1, -1, 1, 0, -1};

static inline uint hash32(uint x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}

static inline uint rnd(uint seed, uint pid, uint t, uint s) {
    return hash32(seed ^ hash32(pid * 0x9E3779B9U + hash32(t * 0x85EBCA6BU + s)));
}

// the label of a cell of one layer, 0 outside the grid
static inline int label_at(device const int* yr, int z, int yy, int xx, int Y, int X) {
    if (yy < 0 || yy >= Y || xx < 0 || xx >= X) return 0;
    return yr[(z * Y + yy) * X + xx];
}

// 1 when a cell of one layer is inside the grid and filled
static inline int filled_at(device const int* tr, int z, int yy, int xx, int Y, int X) {
    if (yy < 0 || yy >= Y || xx < 0 || xx >= X) return 0;
    return tr[(z * Y + yy) * X + xx] >= 0 ? 1 : 0;
}

// The flags of the filled cell (z, yy, xx): bit 0 hot, bit 1 fail (module docstring). own_half.x is the local energy
// of the cell's own label and own_half.y half of its pair costs with filled neighbours: the board energy is the sum
// of (x - y) over the filled cells.
static inline int settle(device const int* yr, device const int* tr, device const float* W8, device const float* B,
                         device const int* bad8, device const int* badB, int z, int yy, int xx, int Y, int X,
                         int rule, float focus_at, float thr, thread float2& own_half) {
    int m = (z * Y + yy) * X + xx;
    int s = tr[m];
    int a0 = yr[m];
    float L[NC];
    for (int a = 0; a < NC; a++) L[a] = B[s * NC + a];
    float half_sum = 0.0f;
    int bad = badB[s * NC + a0] != 0 ? 1 : 0;
    for (int k = 0; k < 8; k++) {
        int c = label_at(yr, z, yy + DY[k], xx + DX[k], Y, X);
        device const float* w = W8 + k * NC * NC + c;
        for (int a = 0; a < NC; a++) L[a] += w[a * NC];
        if (filled_at(tr, z, yy + DY[k], xx + DX[k], Y, X) != 0) half_sum += 0.5f * w[a0 * NC];
        if (bad8[(k * NC + a0) * NC + c] != 0) bad = 1;
    }
    float own = L[a0];
    float low = 3.0e38f;
    for (int a = 1; a < NC; a++) { if (a != a0 && L[a] < low) low = L[a]; }
    float margin = low - own;
    own_half = float2(own, half_sum);
    int hot = 0;
    int fail = 0;
    if (rule == 0) { hot = own > focus_at ? 1 : 0; }
    else if (rule == 1) { hot = own > focus_at ? 1 : 0; fail = bad; }
    else if (rule == 2) { hot = margin < 0.0f ? 1 : 0; fail = margin < thr ? 1 : 0; }
    else { hot = margin < 0.0f ? 1 : 0; fail = hot; }
    return hot | (fail << 1);
}

kernel void moves(device int* y, device const int* tok, device int* flag, device int* found, device float* energy,
                  device const float* W8, device const float* B, device const int* bad8, device const int* badB,
                  constant int& M, constant int& Y, constant int& X, constant int& nmoves, constant int& move0,
                  constant uint& seed, constant float& temp, constant int& rule, constant float& focus_at,
                  constant float& thr,
                  uint pid [[thread_position_in_grid]]) {
    if (found[pid] != -1) return;
    device int* yr = y + pid * uint(M);
    device const int* tr = tok + pid * uint(M);
    device int* fr = flag + pid * uint(M);
    int YX = Y * X;
    int nhot = 0;
    int nfail = 0;
    int nfilled = 0;
    float E = 0.0f;
    float2 oh;
    for (int m = 0; m < M; m++) {
        int f = 0;
        if (tr[m] >= 0) {
            int z = m / YX; int r = m - z * YX; int yy = r / X;
            f = settle(yr, tr, W8, B, bad8, badB, z, yy, r - yy * X, Y, X, rule, focus_at, thr, oh);
            E += oh.x - oh.y;
            nfilled += 1;
        }
        fr[m] = f;
        nhot += f & 1;
        nfail += (f >> 1) & 1;
    }
    for (int i = 0; i < nmoves; i++) {
        uint t = uint(move0 + i);
        bool stop = rule == 0 ? (E <= thr) : (nfail == 0);
        if (stop) { found[pid] = int(t); energy[pid] = E; return; }
        int pool = nhot > 0 ? nhot : nfilled;
        int r = int(rnd(seed, pid, t, 0U) % uint(pool));
        int cell = 0;
        for (int m = 0; m < M; m++) {
            bool in_pool = nhot > 0 ? ((fr[m] & 1) != 0) : (tr[m] >= 0);
            if (in_pool) { if (r == 0) { cell = m; break; } r -= 1; }
        }
        int z = cell / YX; int rem = cell - z * YX; int yy = rem / X; int xx = rem - yy * X;
        int s = tr[cell];
        float L[NC];
        for (int a = 0; a < NC; a++) L[a] = B[s * NC + a];
        for (int k = 0; k < 8; k++) {
            int c = label_at(yr, z, yy + DY[k], xx + DX[k], Y, X);
            device const float* w = W8 + k * NC * NC + c;
            for (int a = 0; a < NC; a++) L[a] += w[a * NC];
        }
        int best = 1;
        if (temp > 0.0f) {
            float top = -3.0e38f;
            for (int a = 1; a < NC; a++) {
                float u = float(rnd(seed, pid, t, uint(1 + a)) >> 8) * (1.0f / 16777216.0f);
                float score = -L[a] / temp - log(-log(max(u, 1.0e-12f)));
                if (score > top) { top = score; best = a; }
            }
        } else {
            float low = L[1];
            for (int a = 2; a < NC; a++) { if (L[a] < low) { low = L[a]; best = a; } }
        }
        E += L[best] - L[yr[cell]];
        yr[cell] = best;
        for (int k = -1; k < 8; k++) {
            int y2 = yy + (k < 0 ? 0 : DY[k]);
            int x2 = xx + (k < 0 ? 0 : DX[k]);
            if (y2 < 0 || y2 >= Y || x2 < 0 || x2 >= X) continue;
            int m2 = (z * Y + y2) * X + x2;
            if (tr[m2] < 0) continue;
            int f = settle(yr, tr, W8, B, bad8, badB, z, y2, x2, Y, X, rule, focus_at, thr, oh);
            nhot += (f & 1) - (fr[m2] & 1);
            nfail += ((f >> 1) & 1) - ((fr[m2] >> 1) & 1);
            fr[m2] = f;
        }
    }
    bool stop = rule == 0 ? (E <= thr) : (nfail == 0);
    if (stop) found[pid] = move0 + nmoves;
    energy[pid] = E;
}
"""

_LIB = None


def library():
    """The compiled kernel library (compiled once per process)."""
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(SOURCE)
    return _LIB


def _gpu(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).to("mps")


def search_chains(K, B, st: dict, tok: np.ndarray, dims, rule: str, temperature: float, cap: int, seed: int,
                  chains: int = 1, tabs: dict = None, theta: float = 0.45, chunk: int = 256, y0=None):
    """The search on the GPU, one row of tok [h, m] int32 per chain; rows come in groups of `chains` per design and
    a design's other chains are cancelled once one of them stops (after the launch in which it stops).
    Returns (y [h, m] int32, found [h] int32, energy [h] float32, seconds in the kernel launches)."""
    Z, Y, X = dims
    h, m = tok.shape
    assert m == Z * Y * X and h % chains == 0 and rule in RULE_ID
    lib = library()
    y = _gpu((BC.init_labels(tok, seed) if y0 is None else y0).astype(np.int32).ravel())
    tok_g = _gpu(tok.astype(np.int32).ravel())
    flag = torch.zeros(h * m, dtype=torch.int32, device="mps")
    found = torch.full((h,), -1, dtype=torch.int32, device="mps")
    energy = torch.zeros(h, dtype=torch.float32, device="mps")
    Kn = K.float().numpy()
    W8, Bg = _gpu(np.concatenate([Kn, Kn.transpose(0, 2, 1)]).ravel()), _gpu(B.float().numpy().ravel())
    zero8, zeroB = np.zeros((8, BC.C, BC.C), np.int32), np.zeros((2, BC.C), np.int32)
    bad8 = _gpu((tabs["bad8"] if tabs is not None else zero8).astype(np.int32).ravel())
    badB = _gpu((tabs["badB"] if tabs is not None else zeroB).astype(np.int32).ravel())
    temp = 0.0 if rule == "fixed" else float(temperature * st["hbar"])
    focus_at = float(st["ebar"] + theta * st["hbar"])
    thr = {"tau": st["tau"] + 1e-3, "local": st["delta"] - 1e-3}.get(rule, 0.0)
    torch.mps.synchronize()
    seconds, move0 = 0.0, 0
    while True:
        n = min(chunk, cap - move0)
        t0 = time.perf_counter()
        lib.moves(y, tok_g, flag, found, energy, W8, Bg, bad8, badB, m, Y, X, n, move0, seed, temp, RULE_ID[rule],
                  focus_at, float(thr), threads=h)
        f = found.cpu().numpy()                                   # waits for the launch
        seconds += time.perf_counter() - t0
        move0 += n
        done = (f >= 0).reshape(-1, chains).any(1)
        if bool(done.all()):
            break
        cancel = (f == -1) & np.repeat(done, chains)
        if cancel.any():
            f = f.copy()
            f[cancel] = -2
            found.copy_(_gpu(f))
        if move0 >= cap:
            break
    return y.cpu().numpy().reshape(h, m), found.cpu().numpy(), energy.cpu().numpy(), seconds


def search(K, B, st: dict, tok: np.ndarray, dims, rule: str, temperature: float, chains: int, cap: int, seed: int,
           tabs: dict = None, theta: float = 0.45, chunk: int = 256) -> dict:
    """`paper_rules.search` on the GPU, for designs tok [n, m]: labels [n, m] (the chain that stopped first, else the
    chain with the lowest energy at the end), stopped [n], moves [n], energy [n], and `confirmed` [n]: the stop holds
    when the labels are judged again in float64 by `paper_rules.passes_rule`. Also gpu_seconds."""
    n = len(tok)
    t = np.repeat(tok, chains, 0)
    y, found, energy, seconds = search_chains(K, B, st, t, dims, rule, temperature, cap, seed, chains, tabs, theta, chunk)
    f, e = found.reshape(n, chains), energy.reshape(n, chains)
    done = (f >= 0).any(1)
    chain = np.where(done, np.where(f >= 0, f, 1 << 30).argmin(1), e.argmin(1))
    idx = np.arange(n) * chains + chain
    labels = y[idx]
    confirmed = np.zeros(n, bool)
    if done.any():
        confirmed[done] = PR.passes_rule(K, B, st, tok[done], dims, labels[done], rule, tabs)
    return {"labels": labels, "stopped": done, "moves": np.where(done, f[np.arange(n), chain], cap),
            "energy": energy[idx], "confirmed": confirmed, "gpu_seconds": seconds}
