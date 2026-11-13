"""The search of brick_cram.py as a custom Metal kernel for Apple GPUs, compiled with torch.mps.compile_shader.

One GPU thread runs one chain: it owns one row of the label buffer `y`, of the token buffer `tok` and of the flag
buffer `viol`, so no two threads write the same memory. A launch makes `nmoves` moves of every chain that has not
stopped. The moves are those of `brick_cram.search_reference`, with the same counter-hash random numbers.

One move of a chain:
  1. stop if no cell has a violated term (found[chain] = moves made so far);
  2. pick the r-th flagged cell, r = hash(seed, chain, move, 0) mod (number of flagged cells);
  3. local energy of each label there: L[a] = B[token, a] + sum over the 8 neighbours k of W8[k, a, label of k];
  4. the new label: argmax over a of -L[a] / temp + Gumbel noise (temp 0: the lowest L[a], the first on a tie);
  5. recompute the flags of the cell and of its neighbours.

`found`: -1 running, -2 cancelled by the host (another chain of its design stopped), >= 0 the moves made at the stop.
"""
import time

import numpy as np
import torch

import brick_cram as BC

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

// 1 when the cell is filled and its input term or one of its 8 pair terms is above its threshold
static inline int violated(device const int* yr, device const int* tr, device const int* bad8, device const int* badB,
                           int z, int yy, int xx, int Y, int X) {
    int m = (z * Y + yy) * X + xx;
    int s = tr[m];
    if (s < 0) return 0;
    int a = yr[m];
    if (badB[s * NC + a] != 0) return 1;
    for (int k = 0; k < 8; k++) {
        int c = label_at(yr, z, yy + DY[k], xx + DX[k], Y, X);
        if (bad8[(k * NC + a) * NC + c] != 0) return 1;
    }
    return 0;
}

kernel void moves(device int* y, device const int* tok, device int* viol, device int* found,
                  device const float* W8, device const float* B, device const int* bad8, device const int* badB,
                  constant int& M, constant int& Y, constant int& X, constant int& nmoves, constant int& move0,
                  constant uint& seed, constant float& temp,
                  uint pid [[thread_position_in_grid]]) {
    if (found[pid] != -1) return;
    device int* yr = y + pid * uint(M);
    device const int* tr = tok + pid * uint(M);
    device int* vr = viol + pid * uint(M);
    int YX = Y * X;
    int nflag = 0;
    for (int m = 0; m < M; m++) {
        int z = m / YX; int r = m - z * YX; int yy = r / X;
        int v = violated(yr, tr, bad8, badB, z, yy, r - yy * X, Y, X);
        vr[m] = v;
        nflag += v;
    }
    for (int i = 0; i < nmoves; i++) {
        uint t = uint(move0 + i);
        if (nflag == 0) { found[pid] = int(t); return; }
        int r = int(rnd(seed, pid, t, 0U) % uint(nflag));
        int cell = 0;
        for (int m = 0; m < M; m++) {
            if (vr[m] != 0) { if (r == 0) { cell = m; break; } r -= 1; }
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
        yr[cell] = best;
        for (int k = -1; k < 8; k++) {
            int y2 = yy + (k < 0 ? 0 : DY[k]);
            int x2 = xx + (k < 0 ? 0 : DX[k]);
            if (y2 < 0 || y2 >= Y || x2 < 0 || x2 >= X) continue;
            int m2 = (z * Y + y2) * X + x2;
            int v = violated(yr, tr, bad8, badB, z, y2, x2, Y, X);
            nflag += v - vr[m2];
            vr[m2] = v;
        }
    }
    if (nflag == 0) found[pid] = move0 + nmoves;
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


def search_gpu(tabs: dict, tok: np.ndarray, dims, max_moves: int, temp: float, seed: int, chains: int = 1,
               chunk: int = 256, y0=None):
    """The search on the GPU. tok [h, m] int32, one row per chain; rows come in groups of `chains` per design, and a
    design's other chains are cancelled once one of them stops. max_moves 0 makes one launch of no moves, so a chain
    that starts with no violated term reports 0, as the reference does.
    Returns (y [h, m] int32, found [h] int32 (see the module docstring), seconds in the kernel launches)."""
    Z, Y, X = dims
    h, m = tok.shape
    assert m == Z * Y * X and h % chains == 0
    lib = library()
    y = _gpu((BC.init_labels(tok, seed) if y0 is None else y0).astype(np.int32).ravel())
    tok_g = _gpu(tok.astype(np.int32).ravel())
    viol = torch.zeros(h * m, dtype=torch.int32, device="mps")
    found = torch.full((h,), -1, dtype=torch.int32, device="mps")
    W8, B = _gpu(tabs["W8"].astype(np.float32).ravel()), _gpu(tabs["B"].astype(np.float32).ravel())
    bad8, badB = _gpu(tabs["bad8"].astype(np.int32).ravel()), _gpu(tabs["badB"].astype(np.int32).ravel())
    torch.mps.synchronize()
    seconds, move0 = 0.0, 0
    while True:
        n = min(chunk, max_moves - move0)
        t0 = time.perf_counter()
        lib.moves(y, tok_g, viol, found, W8, B, bad8, badB, m, Y, X, n, move0, seed, float(temp), threads=h)
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
        if move0 >= max_moves:
            break
    return y.cpu().numpy().reshape(h, m), found.cpu().numpy(), seconds


def moves_per_second(dims, chains: int, nmoves: int = 256, seed: int = 0, filled: float = 0.5, repeats: int = 5) -> dict:
    """Throughput of the kernel on `chains` chains that never stop (every term is flagged as violated): the median of
    `repeats` launches of `nmoves` moves, after a warm-up. The time includes waiting for the launch and reading the
    stop flags back."""
    Z, Y, X = dims
    m = Z * Y * X
    rng = np.random.default_rng(seed)
    tok = np.where(rng.random((chains, m)) < filled, 0, -1).astype(np.int32)
    tok[:, 0] = 0
    tabs = BC.rule_tables()
    tabs = {**tabs, "bad8": np.ones_like(tabs["bad8"]), "badB": np.ones_like(tabs["badB"])}
    search_gpu(tabs, tok, dims, 8, 0.5, seed)                      # warm-up at this size
    times = []
    for _ in range(repeats):
        _, found, seconds = search_gpu(tabs, tok, dims, nmoves, 0.5, seed, chunk=nmoves)
        assert bool((found == -1).all())
        times.append(seconds)
    med = float(np.median(times))
    return {"chains": chains, "cells": m, "moves": chains * nmoves, "seconds_median": med,
            "seconds_min": min(times), "seconds_max": max(times), "moves_per_second": chains * nmoves / med}
