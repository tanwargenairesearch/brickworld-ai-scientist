"""VISUALIZATION ONLY: the energy's search, explained and shown in detail (about four minutes). Nothing here feeds
a score or a selection, and no dev or final design is read.

Scenes
  1. the task, turned into a labelling puzzle;
  2. the 25 labels a cell can carry;
  3. the energy: 2,750 costs counted from solved designs, and what "breaks a rule" means;
  4. the start of the search: a random label on every cell;
  5. one move in detail (the first move and the last move of the chain shown);
  6. the whole search, move by move, in every layer;
  7. the stop, the labels read as bricks, and the build in the frozen world;
  8. why the stop can be trusted, with the recorded numbers;
  9. the same search with the paper's own fit and each of the paper's stopping rules;
 10. the search on the GPU;
 11. what this shows and what it doesn't.

Rules, fixed before the design or its search was looked at:
- Design: the tuning designs of the 10x10x6 world (run_check.tuning_designs, 256 designs), sorted by number of
  cells with ties by index; the one in the middle (position 128 of 0..255). If the counted fit could not finish
  it, the next position would be taken and the video would say so. A tuning design, so that the runs with the
  paper's fit touch no dev design.
- Search shown: the frozen counted fit and settings (learned_settings.json), on this design alone (4 chains,
  seed 0). It is an illustration: the scored runs search whole batches, with other random numbers per chain.
  `collect` steps brick_cram.search_reference one move at a time and stops unless its states equal the
  reference's and the chain shown ends where the Metal kernel ends.
- The two moves of scene 5: the first and the last move of the chain that stopped first.
- Scene 9, boards: the settings chosen by run_paper_rules_wide.py for each rule, the setting of
  run_paper_rules_safe.py and the counted fit, each searched on this design alone through the Metal kernel of
  metal_paper_search.py. Cells are marked with the world's true rules (brick_cram.rule_tables), for display only.
  `collect` stops unless the counted fit's run through that kernel ends on the very board of the search shown
  (with costs of 0 and 1 the paper's focus rule picks exactly the cells that break a rule).
- Every measured number on screen is read from a results file by `collect` and saved to search_data.json.
- No frame may hold a word of BANNED, nor an em or en dash (checked on every string drawn).

usage (project root):
  .venv/bin/python -I -B energy_check/video/make_search_video.py collect   # needs an Apple GPU, for a few seconds
  .venv/bin/python -I -B energy_check/video/make_search_video.py stills    # PNGs in video/search_stills/
  .venv/bin/python -I -B energy_check/video/make_search_video.py video     # video/energy_search.mp4
"""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ENERGY = HERE.parent
ROOT = ENERGY.parent
sys.path.insert(0, str(HERE))

import make_compare_video as MC  # noqa: E402  (the drawing helpers of the first video)

DATA, TRACE, OUT = HERE / "search_data.json", HERE / "search_detail.npz", HERE / "energy_search.mp4"
STILLS = HERE / "search_stills"
DESIGN_RULE = ("the 256 tuning designs of the 10×10×6 world, sorted by number of cells (ties by index): "
               "the one in the middle")
PAPER = "arXiv:2611.17309"


# ---------------------------------------------------------------------------------------------- collect
def trace(BC, np, tabs, tok, dims, temp, seed, max_moves):
    """brick_cram.search_reference on the chains of one design, one move at a time, until the first chain has no
    violated term. Returns the labels and flags after every move, and for every move of every chain the cell
    picked, its old and new label and the 26 local energies."""
    nbr = BC.neighbours(*dims)
    y = BC.init_labels(tok, seed)
    h = y.shape[0]
    idx = np.arange(h)
    W8, B = tabs["W8"], tabs["B"]
    streams = (1 + np.arange(1, BC.C))[None, :]
    ys, fl, cells, olds, news, Ls = [y.copy()], [BC.violations(y, tok, tabs, nbr)], [], [], [], []
    for t in range(max_moves):
        viol = fl[-1]
        nflag = viol.sum(1)
        if (nflag == 0).any():
            break
        r = (BC.rnd(seed, idx, t, 0) % nflag.astype(np.uint32)).astype(np.int64)
        cell = (np.cumsum(viol, 1) > r[:, None]).argmax(1)
        nb = nbr[cell]
        ynb = np.where(nb >= 0, y[idx[:, None], np.clip(nb, 0, None)], 0)
        L = B[tok[idx, cell]].copy()
        for k in range(8):
            L += W8[k][:, ynb[:, k]].T
        u = (BC.rnd(seed, idx[:, None], t, streams) >> np.uint32(8)).astype(np.float32) * np.float32(1.0 / 16777216.0)
        g = -np.log(-np.log(np.maximum(u, np.float32(1e-12))))
        new = 1 + (-L[:, 1:] / np.float32(temp) + g).argmax(1)
        cells.append(cell)
        olds.append(y[idx, cell].copy())
        news.append(new)
        Ls.append(L.copy())
        y[idx, cell] = new
        ys.append(y.copy())
        fl.append(BC.violations(y, tok, tabs, nbr))
    return {"traj": np.stack(ys), "flags": np.stack(fl), "cell": np.stack(cells), "old": np.stack(olds),
            "new": np.stack(news), "L": np.stack(Ls)}


def collect():
    import numpy as np
    import torch

    torch.set_num_threads(4)
    sys.path.insert(0, str(ENERGY))
    SB = MC.module("search_build", MC.VIZ2B / "search_build.py")          # registers W; Stage 1's replay
    R = SB.R
    import metal_paper_search as MP
    import run_learned as RL
    import run_paper_rules as RP
    import run_paper_rules_wide as RW

    BC, BE, MS, RC = RL.BC, RL.BE, RL.MS, RL.RC
    res = lambda name: json.loads((RL.RESULTS / name).read_text())  # noqa: E731
    spec, _, hashes = RC.worlds()["w"]
    designs = RC.tuning_designs(spec, hashes)
    ncell = designs.flatten(1).sum(1)
    order = sorted(range(len(designs)), key=lambda i: (int(ncell[i]), i))
    dims, m = (spec.Z, spec.Y, spec.X), spec.Z * spec.Y * spec.X
    s = json.loads(RL.SETTINGS.read_text())["setting"]
    dev = res("learned_dev.json")
    assert s == dev["setting"]
    tabs, fit = RL.fitted_tables(s["n_train"])
    for k in ("filled_cells", "parameters", "nonzero_parameters", "forbidden_pair_terms_flagged",
              "allowed_pair_terms_flagged", "forbidden_input_terms_flagged"):
        assert fit[k] == dev["fit"][k], k
    RL.warm_up()
    nbr = BC.neighbours(*dims)

    # the design, and its search one move at a time
    pos = len(order) // 2
    while True:
        di = order[pos]
        target = designs[di]
        tok = np.repeat(BC.tokens(target[None]).reshape(1, m).numpy().astype(np.int32), s["chains"], 0)
        y_gpu, f_gpu, _ = MS.search_gpu(tabs, tok, dims, s["max_moves"], s["temp"], s["seed"], chains=s["chains"])
        if (f_gpu >= 0).any():
            break
        pos += 1
    tr = trace(BC, np, tabs, tok, dims, s["temp"], s["seed"], s["max_moves"])
    moves = len(tr["cell"])
    chain = int(np.where(f_gpu >= 0, f_gpu, 1 << 30).argmin())
    y_ref, f_ref = BC.search_reference(tabs, tok, dims, moves, s["temp"], s["seed"])
    assert (y_ref == tr["traj"][-1]).all(), "the trace differs from the reference search"
    assert f_ref[chain] == moves == f_gpu[chain] and (np.delete(f_ref, chain) == -1).all()
    assert (y_gpu[chain] == tr["traj"][-1, chain]).all(), "the trace differs from the Metal kernel"
    assert not tr["flags"][-1, chain].any() and tr["flags"][:-1].any(2).all()
    assert ((tr["traj"][1:] != tr["traj"][:-1]).sum(2) <= 1).all()

    def built(labels_flat):
        labels = torch.from_numpy(labels_flat.reshape(1, *dims)).long()
        board = BC.decode(labels)
        acts = BE.bricks_to_actions(board, spec.done_action)
        occ, illegal = BE.build(acts, target[None], spec)
        exact = bool(RL.check(occ.numpy(), target[None].numpy())["exact"][0])
        bricks, end, occ2 = R.replay(acts[0].tolist(), target, spec)
        return {"bricks": [{"cells": [list(map(int, c)) for c in b["cells"]], "wrong": bool(b["wrong"])} for b in bricks],
                "exact": exact, "illegal": int(illegal), "end": end, "n_bricks": int(board.sum()),
                "missing": int((target & ~occ2).sum()), "extra": int((occ2 & ~target).sum())}

    comp = np.zeros((moves, BC.C, 9), dtype=np.float32)
    for t in range(moves):
        c = int(tr["cell"][t, chain])
        ynb = np.where(nbr[c] >= 0, tr["traj"][t, chain][np.clip(nbr[c], 0, None)], 0)
        comp[t, :, 0] = tabs["B"][tok[chain, c]]
        for k in range(8):
            comp[t, :, 1 + k] = tabs["W8"][k][:, ynb[k]]
        assert np.allclose(comp[t].sum(1), tr["L"][t, chain])
    build = built(tr["traj"][-1, chain])
    assert build["exact"] and build["illegal"] == 0 and build["end"] == "pressed DONE"

    # example cost terms, with how often each occurs in the training labellings
    lab_t, tok_t = RL.training_set(s["n_train"])
    cK, cB = BC.class_counts(lab_t, tok_t)
    K, _, _ = BC.count_fit(lab_t, tok_t)
    L12a, L12b, L11 = BC.label_of(1, 0, 0), BC.label_of(1, 1, 0), BC.label_of(0, 0, 0)
    examples = [
        {"kind": "pair", "a": L12a, "c": L12b, "count": int(cK[0, L12a, L12b]), "cost": float(K[0, L12a, L12b])},
        {"kind": "pair", "a": L12a, "c": L11, "count": int(cK[0, L12a, L11]), "cost": float(K[0, L12a, L11])},
        {"kind": "pair", "a": L11, "c": L11, "count": int(cK[0, L11, L11]), "cost": float(K[0, L11, L11])},
        {"kind": "input", "token": 1, "a": L11, "count": int(cB[1, L11]), "cost": float(tabs["B"][1, L11])},
        {"kind": "input", "token": 0, "a": L11, "count": int(cB[0, L11]), "cost": float(tabs["B"][0, L11])},
    ]
    for e in examples:
        assert (e["count"] == 0) == (e["cost"] == 1.0), e

    # the paper's fit with each of its stopping rules, on this design
    wide, safe = res("paper_rules_wide_fresh.json"), res("paper_rules_safe.json")
    chosen = res("paper_rules_wide_settings.json")["chosen"]
    fits = RW.load_fits()
    tok1, true = tok[:1], BC.rule_tables()
    runs = [{"key": r, **{k: chosen[r][k] for k in ("fit", "rule", "temperature")}} for r in ("tau", "terms", "local", "fixed")]
    runs.append({"key": "light", **{k: safe["setting"][k] for k in ("fit", "rule", "temperature")}})
    runs.append({"key": "count", **RW.CONTROL})
    boards, blabels, bflags = [], [], []
    info = res("paper_rules_wide_fits.json")["fits"]
    for run in runs:
        f = fits[run["fit"]]
        out = MP.search(f[0], f[1], f[2], tok1, dims, run["rule"], run["temperature"], RW.CHAINS, RW.CAP,
                        RW.SEARCH_SEED, tabs=RP.tables(f, run["rule"]))
        lab = out["labels"][:1]
        flag = BC.violations(lab, tok1, true, nbr)[0]
        b = built(lab[0])
        stopped = bool(out["stopped"][0] and out["confirmed"][0])
        agg_src = safe if run["key"] == "light" else wide
        agg = next(a for a in agg_src["summary"] if all(a[k] == run[k] for k in ("fit", "rule", "temperature")))
        boards.append({**run, "stopped": stopped, "moves": int(out["moves"][0]), "exact": b["exact"],
                       "cells_breaking_a_true_rule": int(flag.sum()), "missing": b["missing"], "extra": b["extra"],
                       "fit_info": ({k: info[run["fit"]][k] for k in ("lr", "updates", "l1")} if run["fit"] in info else None),
                       "fresh": {k: agg[k] for k in ("n", "exact", "stopped_and_exact", "wrong_stops",
                                                     "stopped_valid_over_step_limit", "not_stopped")}})
        blabels.append(lab[0])
        bflags.append(flag)
        print(json.dumps(boards[-1]))
    # with counted costs the paper's focus rule picks exactly the cells that break a rule: the same search
    assert boards[-1]["key"] == "count" and (blabels[-1] == tr["traj"][-1, chain]).all() and boards[-1]["moves"] == moves
    safe_control = next(a for a in safe["summary"] if a["fit"] == "count")
    tune_rows = res("paper_rules_wide_tune.json")["rows"]

    wrow = next(r for r in dev["rows"] if r["world"] == "w")
    fresh = res("learned_fresh.json")
    fresh_runs = {}
    for r_ in fresh["rows"]:
        a = fresh_runs.setdefault((r_["fit_seed"], r_["search_seed"]), [0, 0, 0, 0])
        a[0] += r_["exact"]
        a[1] += r_["n"]
        a[2] += r_["stopped_but_not_exact"] - r_["stopped_over_step_limit"]
        a[3] += r_["not_stopped"]
    speed = res("kernel_speed.json")
    fitsize = {}
    for r_ in res("learned_fitsize.json")["rows"]:
        per = fitsize.setdefault(str(r_["n_train"]), {})
        per[str(r_["fit_seed"])] = per.get(str(r_["fit_seed"]), 0) + r_["impossible"]
    fitsize = {k: sorted(v.values()) for k, v in fitsize.items()}
    zs = target.nonzero()[:, 0]
    data = {
        "world": {"X": spec.X, "Y": spec.Y, "Z": spec.Z, "max_steps": spec.max_steps},
        "design": {"rule": DESIGN_RULE, "index": int(di), "position": pos, "of": len(order), "cells": int(ncell[di]),
                   "layers": int(zs.max()) + 1, "target": target.int().tolist(),
                   "resting": int((tok[0] == 0).sum()), "over_empty": int((tok[0] == 1).sum())},
        "setting": s, "fit": fit, "fit_seconds": dev["fit"]["fit_seconds"],
        "seen_pair_terms": tabs["seen_pair_terms"], "seen_input_terms": tabs["seen_input_terms"],
        "examples": examples, "chain": chain, "moves": moves, "build": build,
        "dev": {"exact": sum(r["exact"] for r in dev["rows"]), "n": sum(r["n"] for r in dev["rows"]),
                "wrong_stops": sum(r["stopped_but_not_exact"] - r["stopped_over_step_limit"] for r in dev["rows"]),
                "w": {k: wrow[k] for k in ("n", "exact", "moves_max", "gpu_search_seconds")},
                "w_moves": wrow["per_design"]["moves"],
                "w_moves_median": float(np.median(np.array(wrow["per_design"]["moves"])))},
        "fresh": {"n_per_run": next(iter(fresh_runs.values()))[1], "runs": len(fresh_runs),
                  "exact_min": min(v[0] for v in fresh_runs.values()), "exact_max": max(v[0] for v in fresh_runs.values()),
                  "wrong_stops": sum(v[2] for v in fresh_runs.values())},
        "paper": {"boards": boards, "fits_tried": len(info),
                  "fits_usable": len({r["fit"] for r in tune_rows if "skipped" not in r and r["fit"] != "count"}),
                  "temperatures": len(RW.TEMPS), "n_tune": 4 * RW.N_TUNE,
                  "n_fresh": wide["summary"][0]["n"], "light_control": {k: safe_control[k] for k in (
                      "n", "stopped_and_exact", "wrong_stops", "stopped_valid_over_step_limit", "not_stopped")},
                  "light_flagged": safe["flagged_terms"]},
        "fitsize_impossible": fitsize,
        "speed": {"metal": speed["metal_kernel"], "numpy": speed["numpy_reference_4_threads"]},
    }
    DATA.write_text(json.dumps(data))
    np.savez_compressed(TRACE, traj=tr["traj"].astype(np.int8), flags=tr["flags"], cell=tr["cell"].astype(np.int32),
                        old=tr["old"].astype(np.int8), new=tr["new"].astype(np.int8), L=tr["L"].astype(np.float32),
                        tok=tok[0].astype(np.int8), comp=comp, blabels=np.stack(blabels).astype(np.int8), bflags=np.stack(bflags))
    d = data["design"]
    print("design", d["index"], "position", d["position"], "cells", d["cells"], "layers", d["layers"],
          "| chain", chain, "moves", moves, "bricks", build["n_bricks"])
    print("flagged cells by move, chain shown:", tr["flags"][:, chain].sum(1).tolist())
    print("examples:", json.dumps(examples))
    for t in (0, moves - 1):
        c = int(tr["cell"][t, chain])
        print("move", t + 1, "cell (z, y, x)", divmod(c, spec.Y * spec.X)[0], divmod(c % (spec.Y * spec.X), spec.X),
              "old", int(tr["old"][t, chain]), "new", int(tr["new"][t, chain]),
              "L", [round(float(v), 1) for v in tr["L"][t, chain][1:]])


# ---------------------------------------------------------------------------------------------- drawing
W, H, FPS = MC.W, MC.H, MC.FPS
INK, INK2, MUTED, GRIDC, SURF, CARD = MC.INK, MC.INK2, MC.MUTED, MC.GRIDC, MC.SURF, MC.CARD
GREEN, RED, GREY, BLUE = MC.ENERGY_COL, MC.BAD_COL, MC.OTHER, MC.PICK_COL
PLATE = (244, 244, 240)
T, para, wrap, new, header, swatch = MC.T, MC.para, MC.wrap, MC.new, MC.header, MC.swatch
KICKER = "Suggestion 2 · an energy · the search in detail"
SHAPES = ("1×1", "1×2", "2×1", "2×2")
BC = None


def load():
    global BC
    import numpy as np
    import brick_cram
    BC = brick_cram
    MC.R = MC.module("r1", ROOT / "arch_rsi" / "viz" / "stage1" / "render_learning_stage1.py")
    V = json.loads(DATA.read_text())
    z = np.load(TRACE)
    V.update({k: z[k] for k in z.files})
    sp = SimpleNamespace(**{k: V["world"][k] for k in "XYZ"})
    V["spec"], V["shape"] = sp, (sp.Z, sp.Y, sp.X)
    V["target"] = np.array(V["design"]["target"], dtype=bool)
    V["tok3"] = V["tok"].reshape(V["shape"])
    V["build"]["bricks"] = [{"cells": [tuple(c) for c in b["cells"]], "wrong": b["wrong"]} for b in V["build"]["bricks"]]
    V["energy"] = {"moves": V["moves"]}                               # the fields MC.chart reads
    V["np"] = np
    assert V["traj"].shape[0] == V["moves"] + 1
    return V


def glyph(d, cx, cy, u, a, on, off):
    """A label as a picture: its brick's shape, this cell's piece filled, and a dot on the brick's support piece."""
    t, p, q = int(BC.TYPE[a]), int(BC.POS[a]), int(BC.ANCHOR[a])
    cells = BC.CELLS[t]
    ny, nx = max(c[0] for c in cells) + 1, max(c[1] for c in cells) + 1
    gap, lw = max(1, round(u * 0.22)), max(1, round(u / 6))
    x0, y0 = cx - (nx * u + (nx - 1) * gap) / 2, cy - (ny * u + (ny - 1) * gap) / 2
    for i, (oy, ox) in enumerate(cells):
        bx, by = x0 + ox * (u + gap), y0 + oy * (u + gap)
        if i == p:
            d.rectangle([bx, by, bx + u, by + u], fill=on)
        else:
            d.rectangle([bx, by, bx + u, by + u], outline=on, width=lw)
        if i == q:
            r = max(1.5, u * 0.2)
            d.ellipse([bx + u / 2 - r, by + u / 2 - r, bx + u / 2 + r, by + u / 2 + r], fill=off if i == p else on)


def group_of(a, y, x):
    """The brick a label claims: (shape, support piece, row and column of the brick's first cell)."""
    t = int(BC.TYPE[a])
    dy, dx = BC.CELLS[t][int(BC.POS[a])]
    return (t, int(BC.ANCHOR[a]), y - dy, x - dx)


def cross(d, cx, cy, r, w=2, col=(255, 255, 255)):
    d.line([(cx - r, cy - r), (cx + r, cy + r)], fill=col, width=w)
    d.line([(cx - r, cy + r), (cx + r, cy - r)], fill=col, width=w)


def tile(d, bx, by, cs, a, bad, over=False, col=None, edges=(True, True, True, True), mark=True):
    """One labelled cell. edges: thick outline on (top, left, right, bottom)."""
    col = col or (RED if bad else GREEN)
    d.rectangle([bx, by, bx + cs, by + cs], fill=col)
    ew, dark, soft = max(2, round(cs / 14)), MC.R.shade(col, 0.5), MC.R.mix(col, (255, 255, 255), 0.3)
    for on, seg in zip(edges, ([(bx, by), (bx + cs, by)], [(bx, by), (bx, by + cs)],
                               [(bx + cs, by), (bx + cs, by + cs)], [(bx, by + cs), (bx + cs, by + cs)])):
        d.line(seg, fill=dark if on else soft, width=ew if on else 1)
    if a:
        glyph(d, bx + cs / 2, by + cs / 2, max(4, round(cs * 0.2)), a, (255, 255, 255), col)
    if bad and mark:
        cross(d, bx + cs * 0.84, by + cs * 0.17, max(2, cs * 0.075), max(2, round(cs / 22)))
    if over:                                                    # the cell hangs over an empty cell
        n, y = 3, by + cs * 0.9
        for i in range(n):
            x = bx + cs * (0.2 + 0.6 * i / n)
            d.line([(x, y), (x + cs * 0.12, y)], fill=(255, 255, 255), width=max(2, round(cs / 22)))


def layer_view(d, V, x0, y0, cs, z, lab=None, flag=None, ring=None, window=None, title=None):
    """One layer from above. lab, flag: [Y, X] arrays (None: the target alone, each filled cell a question mark)."""
    sp, tgt, tok = V["spec"], V["target"][z], V["tok3"][z]
    X, Y = sp.X, sp.Y
    if title:
        T(d, (x0, y0 - 34), title, 18, INK2, bold=True)
    d.rounded_rectangle([x0 - 8, y0 - 8, x0 + X * cs + 8, y0 + Y * cs + 8], radius=10, fill=PLATE)
    for i in range(X + 1):
        d.line([(x0 + i * cs, y0), (x0 + i * cs, y0 + Y * cs)], fill=GRIDC, width=1)
    for j in range(Y + 1):
        d.line([(x0, y0 + j * cs), (x0 + X * cs, y0 + j * cs)], fill=GRIDC, width=1)
    grp = {(y, x): group_of(int(lab[y, x]), y, x) for y in range(Y) for x in range(X) if tgt[y, x]} if lab is not None else {}
    for y in range(Y):
        for x in range(X):
            if not tgt[y, x]:
                continue
            bx, by = x0 + x * cs, y0 + y * cs
            if lab is None:
                tile(d, bx, by, cs, 0, False, col=MC.R.TARGET_COL)
                T(d, (bx + cs / 2, by + cs / 2), "?", round(cs * 0.42), (255, 255, 255), bold=True, anchor="mm")
                continue
            g = grp[(y, x)]
            edges = tuple(grp.get(n) != g for n in ((y - 1, x), (y, x - 1), (y, x + 1), (y + 1, x)))
            tile(d, bx, by, cs, int(lab[y, x]), bool(flag[y, x]), over=tok[y, x] == 1, edges=edges)
    if window:
        y, x = window
        ya, xa, yb, xb = max(0, y - 1), max(0, x - 1), min(Y, y + 2), min(X, x + 2)
        d.rectangle([x0 + xa * cs - 3, y0 + ya * cs - 3, x0 + xb * cs + 3, y0 + yb * cs + 3], outline=INK2, width=2)
    if ring:
        y, x = ring
        d.rectangle([x0 + x * cs - 4, y0 + y * cs - 4, x0 + (x + 1) * cs + 4, y0 + (y + 1) * cs + 4], outline=INK, width=4)


def legend(d, x, y, over=True):
    cs = 22
    tile(d, x, y, cs, 0, True)
    T(d, (x + cs + 10, y + 1), "breaks a rule", 17, INK)
    x += cs + 10 + MC.R.font(17).getlength("breaks a rule") + 30
    tile(d, x, y, cs, 0, False)
    T(d, (x + cs + 10, y + 1), "no rule broken here", 17, INK)
    x += cs + 10 + MC.R.font(17).getlength("no rule broken here") + 30
    if over:
        tile(d, x, y, cs, 0, False, over=True, col=GREY)
        T(d, (x + cs + 10, y + 1), "hangs over an empty cell", 17, INK)
    return x


def arrow(d, x0, y, x1, col=INK2, w=3):
    d.line([(x0, y), (x1 - 4, y)], fill=col, width=w)
    d.polygon([(x1, y), (x1 - 12, y - 7), (x1 - 12, y + 7)], fill=col)


def state(V, k, chain=None):
    """Labels and flags [Z, Y, X] of a chain after k moves."""
    c = V["chain"] if chain is None else chain
    return V["traj"][k, c].reshape(V["shape"]), V["flags"][k, c].reshape(V["shape"])


def iso_cells(d, V, iso, lab, flag):
    cells, group, colour, crossed = [], {}, {}, set()
    for z, y, x in zip(*V["target"].nonzero()):
        c = (int(z), int(y), int(x))
        cells.append(c)
        group[c] = (c[0],) + group_of(int(lab[c]), c[1], c[2])
        colour[c] = RED if flag[c] else GREEN
        if flag[c]:
            crossed.add(c)
    iso.ground(d, footprint=V["target"].any(0))
    MC.draw_cells(d, iso, cells, group, colour, crossed)


def iso_at(V, cx, top, width, height):
    return MC.iso_in(V, cx, top, width, V["design"]["layers"], height)


def footer(d, s):
    T(d, (40, H - 34), s, 15, MUTED)


# ---------------------------------------------------------------------------------------------- scenes
LV = {"cs": 44, "x": (56, 536), "y": 212}          # the two layer views of scenes 4, 6 and 7


def two_layers(d, V, lab=None, flag=None, ring=None):
    """ring: (z, y, x) of the cell changed by the last move."""
    names = ("Layer 1, on the ground", "Layer 2, on top of layer 1")
    for z in range(V["design"]["layers"]):
        layer_view(d, V, LV["x"][z], LV["y"], LV["cs"], z, None if lab is None else lab[z],
                   None if flag is None else flag[z], ring=ring[1:] if ring and ring[0] == z else None, title=names[z])


def scene_task(V, stage):
    img, d = new()
    D = V["design"]
    header(d, KICKER, "Building a design is turned into a labelling puzzle",
           "The AI scientist's second suggestion was to drop the network and search with an energy. This video shows "
           "that search step by step, on one design.")
    T(d, (480, 150), "The design to build", 22, INK, bold=True, anchor="ma")
    T(d, (480, 182), f"{D['cells']} filled cells in {D['layers']} layers of a 10×10×6 world", 17, INK2, anchor="ma")
    iso = iso_at(V, 480, 250, 760, 440)
    MC.draw_target(d, iso, V["target"])
    if stage >= 1:
        T(d, (1420, 150), "The same design, layer by layer, from above", 22, INK, bold=True, anchor="ma")
        T(d, (1420, 182), "every filled cell needs a label", 17, INK2, anchor="ma")
        cs = 40
        for z in range(D["layers"]):
            layer_view(d, V, 990 + z * 460, 250, cs, z, title=("Layer 1, on the ground", "Layer 2, on top of layer 1")[z])
    d.line([(40, 730), (W - 40, 730)], fill=GRIDC, width=1)
    y = para(d, (60, 760), "The task: cover the filled cells exactly with bricks of four shapes (1×1, 1×2, 2×1, 2×2), "
                           "each brick resting on the ground or on another brick.", 22, 860, INK)
    para(d, (60, y + 12), "A network places one brick at a time and is trained by reward.", 20, 860, INK2)
    if stage >= 1:
        y = para(d, (1000, 760), "The energy does it differently. It gives every filled cell a label that says which "
                                 "piece of which brick the cell is.", 22, 860, INK)
        if stage >= 2:
            para(d, (1000, y + 12), "If all the labels agree with their neighbours, the bricks can be read off and "
                                    "built. So the search is over labels, not over brick placements.", 20, 860, INK2)
    footer(d, f"Design shown: {D['rule']}. Chosen before its search was looked at. An illustration, not a score.")
    return img


def label_tile(d, x, y, cs, a, col=GREEN):
    tile(d, x, y, cs, a, False, col=col)


def scene_labels(V, stage):
    img, d = new()
    header(d, KICKER, "A cell can carry one of 25 labels",
           "A label says three things: the brick's shape, which piece of the brick this cell is, and which piece is "
           "the brick's support.")
    cs, top = 58, 250
    groups = [(0, 70, "1×1", "1 label"), (1, 250, "1×2", "2 pieces × 2 supports = 4 labels"),
              (2, 640, "2×1", "2 pieces × 2 supports = 4 labels"), (3, 1030, "2×2", "4 pieces × 4 supports = 16 labels")]
    for t, x0, name, count in groups:
        if stage < 1 and t > 0:
            continue
        T(d, (x0, top - 78), name, 26, INK, bold=True)
        T(d, (x0, top - 42), count, 17, INK2)
        pieces = len(BC.CELLS[t])
        for p in range(pieces):
            for q in range(pieces if t else 1):
                a = BC.label_of(t, p, q)
                label_tile(d, x0 + q * (cs + 12), top + p * (cs + 12), cs, a)
        if t:
            T(d, (x0 + (pieces * (cs + 12) - 12) / 2, top + pieces * (cs + 12) + 4), "support piece: 1st, 2nd, ...", 15, MUTED,
              anchor="ma")
            T(d, (x0 - 10, top + (pieces * (cs + 12) - 12) / 2), "piece", 15, MUTED, anchor="rm")
    # how to read a tile
    x0, y0 = 1420, 160
    d.rounded_rectangle([x0 - 30, y0, W - 40, 590], radius=14, fill=CARD, outline=GRIDC, width=1)
    T(d, (x0, y0 + 22), "HOW TO READ A LABEL", 15, MUTED, bold=True)
    big = 120
    tile(d, x0, y0 + 60, big, BC.label_of(3, 1, 2), False)
    y = para(d, (x0 + big + 26, y0 + 58), "Small squares: the brick's shape, here 2×2.", 18, 300, INK)
    y = para(d, (x0 + big + 26, y + 6), "Filled square: this cell's piece, here top right.", 18, 300, INK)
    para(d, (x0 + big + 26, y + 6), "Dot: the support piece, here bottom left.", 18, 300, INK)
    para(d, (x0, y0 + 268), "The support piece is the one cell of the brick that must rest on the ground or on a "
                            "filled cell. That is how a labelling says that no brick floats.", 18, 440, INK2)
    if stage >= 2:
        d.line([(40, 640), (W - 40, 640)], fill=GRIDC, width=1)
        T(d, (60, 668), "Labels that agree form a brick", 22, INK, bold=True)
        cs2, y1 = 64, 730
        # a 2x2 brick, a 1x2 brick, and two cells that disagree
        for i, (oy, ox) in enumerate(BC.CELLS[3]):
            ed = (oy == 0, ox == 0, ox == 1, oy == 1)
            tile(d, 60 + ox * cs2, y1 + oy * cs2, cs2, BC.label_of(3, i, 2), False, edges=ed)
        para(d, (60, y1 + 2 * cs2 + 14), "Four cells, the four pieces of one 2×2 brick, the same support.", 17, 330, INK2)
        for i in range(2):
            tile(d, 480 + i * cs2, y1 + 32, cs2, BC.label_of(1, i, 0), False, edges=(True, i == 0, i == 1, True))
        para(d, (480, y1 + 2 * cs2 + 14), "Two cells, the two pieces of one 1×2 brick.", 17, 300, INK2)
        tile(d, 900, y1 + 32, cs2, BC.label_of(1, 0, 0), True)
        tile(d, 900 + cs2, y1 + 32, cs2, BC.label_of(0, 0, 0), True)
        para(d, (900, y1 + 2 * cs2 + 14), "The left cell says it is the first piece of a 1×2 brick. Its right "
                                          "neighbour says it is a 1×1 brick. They do not agree, so both break a "
                                          "rule.", 17, 400, INK2)
        para(d, (1420, 668), "Finding a labelling in which every cell agrees with its neighbours, and every support "
                             "piece rests on something, is the whole problem. A brick is then placed wherever a label "
                             "is a brick's first piece.", 20, 440, INK)
    footer(d, "The 25 labels were designed from the world's rules. They are an input to the method, not something it learned.")
    return img


def pair_example(d, x, y, cs, a, c):
    tile(d, x, y, cs, a, False, col=GREY)
    tile(d, x + cs, y, cs, c, False, col=GREY)


def scene_energy(V, stage):
    img, d = new()
    F = V["fit"]
    header(d, KICKER, "The energy: a table of costs, counted from solved designs",
           f"{F['parameters']:,} costs. A term that occurs in the solved designs costs 0. A term that never occurs "
           "costs 1. Nothing else is fitted.")
    # left: which terms exist
    x0, y0, cs = 70, 190, 78
    T(d, (x0, y0 - 34), "WHICH TERMS", 15, MUTED, bold=True)
    for j in range(3):
        for i in range(3):
            bx, by = x0 + i * cs, y0 + j * cs
            d.rectangle([bx, by, bx + cs, by + cs], fill=PLATE, outline=GRIDC, width=2)
    d.rectangle([x0 + cs, y0 + cs, x0 + 2 * cs, y0 + 2 * cs], fill=GREEN)
    ccx, ccy = x0 + 1.5 * cs, y0 + 1.5 * cs
    for dr, dc in BC.NB8:
        d.line([(ccx + dc * cs * 0.3, ccy + dr * cs * 0.3), (ccx + dc * cs * 0.82, ccy + dr * cs * 0.82)], fill=INK, width=3)
        d.ellipse([ccx + dc * cs * 0.82 - 6, ccy + dr * cs * 0.82 - 6, ccx + dc * cs * 0.82 + 6, ccy + dr * cs * 0.82 + 6], fill=INK)
    y = para(d, (x0 + 3 * cs + 34, y0 - 4), "Pair terms: a cell's label with the label of each of its 8 neighbours in "
                                           "the same layer, sides and corners.", 19, 310, INK)
    para(d, (x0 + 3 * cs + 34, y + 10), "Input terms: a cell's label with what is under the cell, something or "
                                       "nothing.", 19, 310, INK)
    y = para(d, (x0, y0 + 3 * cs + 26), "4 directions × 26 × 26 pair costs and 2 × 25 input costs, less the 4 "
                                       "empty-with-empty pairs. Empty counts as a 26th label.", 18, 570, INK2)
    para(d, (x0, y + 10), "There is no cost between layers. Layers are tied together only through the input terms.", 18,
         570, INK2)
    # middle: counting
    if stage >= 1:
        x1 = 720
        T(d, (x1, y0 - 34), f"COUNTED ONCE, FROM {F['n_train']:,} SOLVED DESIGNS OF A SMALLER WORLD (5×5×3)", 15, MUTED, bold=True)
        cs2, yy = 54, y0 + 6
        say = lambda e: (f"seen {e['count']:,} times" if e["count"] else "never seen")  # noqa: E731
        ex = V["examples"]
        rows = [(ex[0], "the two pieces of a 1×2 brick, side by side"), (ex[2], "two 1×1 bricks side by side"),
                (ex[1], "first piece of a 1×2 brick, then a 1×1 brick"),
                (ex[4], "a 1×1 brick that rests on something"), (ex[3], "a 1×1 brick over an empty cell")]
        for e, what in rows:
            if e["kind"] == "pair":
                pair_example(d, x1, yy, cs2, e["a"], e["c"])
            else:
                tile(d, x1 + cs2 // 2, yy, cs2, e["a"], False, over=e["token"] == 1, col=GREY)
            T(d, (x1 + 2 * cs2 + 24, yy + 4), what, 19, INK)
            T(d, (x1 + 2 * cs2 + 24, yy + 30), say(e), 16, INK2)
            bad = e["cost"] > 0
            bx = x1 + 620
            d.rounded_rectangle([bx, yy + 8, bx + 96, yy + 46], radius=8, fill=RED if bad else GREEN)
            T(d, (bx + 48, yy + 27), f"cost {e['cost']:g}", 18, (255, 255, 255), bold=True, anchor="mm")
            yy += cs2 + 30
        para(d, (x1, yy + 4), f"The count took {V['fit_seconds'] * 1000:.1f} ms. {F['nonzero_parameters']:,} of the "
                              f"{F['parameters']:,} costs are 1. No reward, no network, no gradient steps.", 18, 720, INK2)
    if stage >= 2:
        x2 = 1500
        d.rounded_rectangle([x2 - 30, y0 - 44, W - 40, 700], radius=14, fill=CARD, outline=GRIDC, width=1)
        T(d, (x2, y0 - 22), "WHAT THE SEARCH USES", 15, MUTED, bold=True)
        y = para(d, (x2, y0 + 8), "Energy of a labelling", 22, 330, INK, bold=True)
        y = para(d, (x2, y + 4), "The sum of its costs: how many never-seen terms it holds.", 19, 330, INK)
        y = para(d, (x2, y + 22), "A cell breaks a rule", 22, 330, INK, bold=True)
        y = para(d, (x2, y + 4), "when one of its own 9 terms costs 1.", 19, 330, INK)
        y = para(d, (x2, y + 22), "Stop", 22, 330, INK, bold=True)
        para(d, (x2, y + 4), "when no cell breaks a rule. The energy is then 0, as it is for every solved design "
                             "the costs were counted from.", 19, 330, INK)
    d.line([(40, 760), (W - 40, 760)], fill=GRIDC, width=1)
    if stage >= 2:
        y = para(d, (60, 790), "The paper this follows learns its costs by gradient descent (pseudolikelihood) and "
                               "stops on a threshold fitted to solved examples. Here the costs are counted instead.", 21, 1780, INK)
        para(d, (60, y + 10), f"The energy form, the 3×3 window with shared costs and the search are the paper's "
                              f"({PAPER}). A later scene shows what happens with the paper's own fit.", 19, 1780, INK2)
    footer(d, "Counts and costs are read from the fitted tables. The training designs come with their bricks, so their labels are known.")
    return img


def chart(d, V, k, box):
    """Cells that break a rule, by search move, for the chains of this design, up to move k."""
    x0, y0, x1, y1 = box
    fl, moves, shown = V["flags"].sum(2), V["moves"], V["chain"]            # [moves + 1, chains]
    top = int(-(-fl.max() // 10) * 10)
    T(d, (x0 - 50, y0 - 46), "Cells that break a rule, by search move", 18, INK, bold=True)
    lx = x1 - 330
    d.line([(lx, y0 - 35), (lx + 26, y0 - 35)], fill=GREEN, width=3)
    T(d, (lx + 34, y0 - 46), "the chain shown", 15, INK2)
    d.line([(lx + 170, y0 - 35), (lx + 196, y0 - 35)], fill=GREY, width=2)
    T(d, (lx + 204, y0 - 46), f"{fl.shape[1] - 1} other chains", 15, INK2)
    X = lambda m: x0 + (x1 - x0) * m / moves  # noqa: E731
    Y = lambda v: y1 - (y1 - y0) * v / top  # noqa: E731
    for v in range(0, top + 1, 10):
        d.line([(x0, Y(v)), (x1, Y(v))], fill=GRIDC, width=1)
        T(d, (x0 - 12, Y(v)), str(v), 14, MUTED, anchor="rm")
    for m in range(0, moves + 1, 10 if moves <= 120 else 50):
        T(d, (X(m), y1 + 8), str(m), 14, MUTED, anchor="ma")
    T(d, ((x0 + x1) / 2, y1 + 28), "search move", 14, MUTED, anchor="ma")
    for c in [c for c in range(fl.shape[1]) if c != shown] + [shown]:
        col, wid = (GREEN, 3) if c == shown else (GREY, 2)
        pts = []
        for m in range(k + 1):
            if m:
                pts.append((X(m), Y(fl[m - 1, c])))
            pts.append((X(m), Y(fl[m, c])))
        if len(pts) > 1:
            d.line(pts, fill=col, width=wid, joint="curve")
        if c == shown:
            cx, cy = X(k), Y(fl[k, c])
            d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=SURF)
            d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=col)


def scene_search(V, k, mode, nb=0):
    """mode: 'start' (scene 4), 'run' (scene 6), 'stop' and 'build' (scene 7)."""
    img, d = new()
    D, moves = V["design"], V["moves"]
    lab, flag = state(V, k)
    n = int(flag.sum())
    titles = {"start": "The search starts from a random label on every cell",
              "run": "Each move relabels one cell that breaks a rule",
              "stop": "No cell breaks a rule: the search stops",
              "build": "The labels are read as bricks and built in the world"}
    subs = {"start": f"{V['setting']['chains']} chains start from different random labels. The one shown is the "
                     "chain that stopped first.",
            "run": "Outlines join the cells that claim to be pieces of the same brick. A ring marks the cell just "
                   "relabelled.",
            "stop": f"After {moves} moves every label agrees with its neighbours, so the energy is 0.",
            "build": "A brick is placed wherever a label is a brick's first piece, lowest layer first. The world's "
                     "own checker compares the result with the design."}
    header(d, KICKER, titles[mode], subs[mode])
    ring = None
    if mode == "run" and k > 0:
        c = int(V["cell"][k - 1, V["chain"]])
        yx = V["spec"].Y * V["spec"].X
        ring = (c // yx, (c % yx) // V["spec"].X, c % V["spec"].X)
    two_layers(d, V, lab, flag, ring)
    legend(d, 56, LV["y"] + 10 * LV["cs"] + 26)
    cx = 1450
    iso = iso_at(V, cx, 230, 800, 400)
    if mode == "build":
        T(d, (cx, 150), "The build", 22, INK, bold=True, anchor="ma")
        MC.draw_build(d, iso, V["target"], V["build"]["bricks"][:nb], GREEN)
        T(d, (cx, 668), f"bricks placed: {nb} of {V['build']['n_bricks']}", 18, INK2, anchor="ma")
    else:
        T(d, (cx, 150), "The same labels, on the design", 22, INK, bold=True, anchor="ma")
        iso_cells(d, V, iso, lab, flag)
        T(d, (cx, 668), f"move {k} of {moves}  ·  cells that break a rule: {n} of {D['cells']}", 18, INK2, anchor="ma")
    d.line([(40, 760), (W - 40, 760)], fill=GRIDC, width=1)
    if mode in ("start", "run", "stop"):
        chart(d, V, k, (110, 850, 1100, 990))
    if mode == "start":
        y = para(d, (1200, 800), f"At the start {n} of the {D['cells']} cells break a rule.", 22, 660, INK)
        para(d, (1200, y + 10), "The search never looks at the whole board. It only repairs cells that break a rule, "
                                "one at a time.", 20, 660, INK2)
    elif mode == "run":
        y = para(d, (1200, 800), "The count goes down, but not at every move: a new label can break a rule "
                                 "somewhere else.", 22, 660, INK)
        para(d, (1200, y + 10), "The four chains run at the same time. The first to reach zero is used and the "
                                "others are dropped.", 20, 660, INK2)
    elif mode == "stop":
        MC.result_line(d, 1530, 800, True, "no cell breaks a rule: stop", 22)
        para(d, (1200, 850), f"This chain needed {moves} moves. The other three chains were still searching and are "
                             "dropped.", 20, 660, INK2)
    else:
        B = V["build"]
        y = para(d, (60, 790), f"Read off: {B['n_bricks']} bricks. They are placed through the same actions a network "
                               "uses, in the same frozen world.", 22, 1000, INK)
        if nb >= B["n_bricks"]:
            MC.result_line(d, 1530, 800, B["exact"], f"built exactly, {B['n_bricks']} bricks", 22)
            para(d, (60, y + 10), "The energy does not try to use few bricks. It only looks for a labelling with no "
                                  f"broken rule. The world allows up to {V['world']['max_steps']} bricks here.", 20, 1000, INK2)
    footer(d, "This design was searched alone, with the frozen settings (4 chains, temperature 0.6, seed 0). The scored "
              "runs search whole batches.")
    return img


def cost_chart(d, V, box, L, temp, show_prob, new_label, old_label):
    """Top: the cost of each of the 25 labels at the picked cell. Bottom: the chance each is drawn."""
    np = V["np"]
    x0, y0, x1, y1 = box
    col = (x1 - x0) / 25
    mid = y0 + (y1 - y0) * 0.44
    cost = L[1:].astype(float)
    top = max(4, int(np.ceil(cost.max())))
    p = np.exp(-(cost - cost.min()) / temp)
    p /= p.sum()
    T(d, (x0, y0 - 34), "Cost of each label at this cell, with the neighbours as they are now", 18, INK, bold=True)
    ch = mid - 40 - y0
    for v in range(top + 1):
        yy = mid - 40 - ch * v / top
        d.line([(x0, yy), (x1, yy)], fill=GRIDC, width=1)
        T(d, (x0 - 10, yy), str(v), 14, MUTED, anchor="rm")
    for i in range(25):
        a = i + 1
        cx = x0 + (i + 0.5) * col
        bar = BLUE if (show_prob and a == new_label) else (120, 119, 114)
        h = ch * cost[i] / top
        if h > 0:
            d.rounded_rectangle([cx - col * 0.3, mid - 40 - h, cx + col * 0.3, mid - 40], radius=3, fill=bar)
        else:
            d.rectangle([cx - col * 0.3, mid - 43, cx + col * 0.3, mid - 40], fill=bar)
        glyph(d, cx, mid - 16, 7, a, INK, SURF)
        if a == old_label:
            d.rectangle([cx - col * 0.46, mid - 34, cx + col * 0.46, mid + 2], outline=MUTED, width=1)
    for name, a0, a1 in (("1×1", 1, 1), ("1×2", 2, 5), ("2×1", 6, 9), ("2×2", 10, 25)):
        xa, xb = x0 + (a0 - 1) * col + 3, x0 + a1 * col - 3
        d.line([(xa, mid + 10), (xb, mid + 10)], fill=MUTED, width=2)
        T(d, ((xa + xb) / 2, mid + 14), name, 14, MUTED, anchor="ma")
    T(d, (x1, mid + 30), "boxed: the label the cell had before this move", 14, MUTED, anchor="ra")
    if show_prob:
        T(d, (x0, mid + 52), f"Chance of each label: proportional to exp(-cost / {temp:g})", 18, INK, bold=True)
        ph, base = y1 - (mid + 100), y1
        pm = max(0.4, float(np.ceil(p.max() * 10) / 10))
        for v in (0, pm / 2, pm):
            yy = base - ph * v / pm
            d.line([(x0, yy), (x1, yy)], fill=GRIDC, width=1)
            T(d, (x0 - 10, yy), f"{v * 100:.0f}%", 14, MUTED, anchor="rm")
        for i in range(25):
            a = i + 1
            cx = x0 + (i + 0.5) * col
            h = max(2, ph * p[i] / pm)
            d.rounded_rectangle([cx - col * 0.3, base - h, cx + col * 0.3, base], radius=3,
                                fill=BLUE if a == new_label else (120, 119, 114))
            if a == new_label or p[i] == p.max():
                T(d, (cx, base - h - 6), f"{p[i] * 100:.0f}%", 15, INK, bold=a == new_label, anchor="md")
        cx = x0 + (new_label - 0.5) * col
        T(d, (cx, base + 8), "drawn", 15, BLUE, bold=True, anchor="ma")
    return p


def zoom(d, V, x0, y0, cs, t, a, lab, flag, z, y, x):
    """The 3x3 window of the picked cell with label `a` on it, and the cost of each of its 9 terms."""
    sp, tgt = V["spec"], V["target"][z]
    comp = V["comp"][t, a]
    for k, (dr, dc) in enumerate(((0, 0),) + BC.NB8):
        yy, xx = y + dr, x + dc
        bx, by = x0 + (1 + dc) * cs, y0 + (1 + dr) * cs
        inside = 0 <= yy < sp.Y and 0 <= xx < sp.X
        if not inside:
            d.rectangle([bx, by, bx + cs, by + cs], fill=SURF, outline=GRIDC, width=1)
            for i in range(1, 6):
                d.line([(bx + i * cs / 6, by), (bx, by + i * cs / 6)], fill=GRIDC, width=1)
        elif k == 0:
            tile(d, bx, by, cs, a, bool(comp.sum() > 0), over=V["tok3"][z, y, x] == 1, mark=False)
        elif tgt[yy, xx]:
            tile(d, bx, by, cs, int(lab[yy, xx]), False, over=V["tok3"][z, yy, xx] == 1, col=GREY)
        else:
            d.rectangle([bx, by, bx + cs, by + cs], fill=PLATE, outline=GRIDC, width=1)
    d.rectangle([x0 + cs - 3, y0 + cs - 3, x0 + 2 * cs + 3, y0 + 2 * cs + 3], outline=INK, width=4)
    ccx, ccy = x0 + 1.5 * cs, y0 + 1.5 * cs
    for k, (dr, dc) in enumerate(BC.NB8):
        bx, by = ccx + dc * cs * 0.5, ccy + dr * cs * 0.5
        bad = comp[1 + k] > 0
        d.ellipse([bx - 19, by - 19, bx + 19, by + 19], fill=SURF)
        d.ellipse([bx - 15, by - 15, bx + 15, by + 15], fill=RED if bad else SURF, outline=MC.R.shade(RED, 0.6) if bad else INK2,
                  width=2)
        T(d, (bx, by), f"{comp[1 + k]:g}", 17, (255, 255, 255) if bad else INK2, bold=True, anchor="mm")
    pairs = int(comp[1:].sum())
    T(d, (x0 + 1.5 * cs, y0 + 3 * cs + 14), f"input cost {comp[0]:g}  +  pair costs {pairs}  =  {comp.sum():g}", 18, INK,
      bold=True, anchor="ma")
    return float(comp.sum())


STEPS = ("Pick a cell", "Score all 25 labels", "Draw the new label", "Write it, recheck 9 cells")


def scene_move(V, t, stage):
    img, d = new()
    sp, D, ch = V["spec"], V["design"], V["chain"]
    moves, temp = V["moves"], V["setting"]["temp"]
    c = int(V["cell"][t, ch])
    z, r = divmod(c, sp.Y * sp.X)
    y, x = divmod(r, sp.X)
    old, new_l = int(V["old"][t, ch]), int(V["new"][t, ch])
    lab0, flag0 = state(V, t)
    lab1, flag1 = state(V, t + 1)
    n0, n1 = int(flag0.sum()), int(flag1.sum())
    which = "the first move" if t == 0 else ("the last move" if t == moves - 1 else f"move {t + 1}")
    header(d, KICKER, f"One move in detail: {which} (move {t + 1} of {moves})",
           "Shown: the first and the last move of the chain that stopped first. The rule was set before the search "
           "was looked at.")
    after = stage >= 3
    lab, flag = (lab1, flag1) if after else (lab0, flag0)
    cs = 52
    layer_view(d, V, 56, 212, cs, z, lab[z], flag[z], ring=(y, x), window=(y, x) if stage >= 1 else None,
               title=("Layer 1, on the ground", "Layer 2, on top of layer 1")[z])
    legend(d, 56, 212 + 10 * cs + 22, over=False)
    # the 3x3 window
    zx, zy, zc = 640, 262, 92
    shown = new_l if stage >= 2 else old
    T(d, (zx + 1.5 * zc, zy - 52), "The cell and its 8 neighbours", 18, INK, bold=True, anchor="ma")
    T(d, (zx + 1.5 * zc, zy - 26), ("with the label drawn" if stage >= 2 else "with the label it has now"), 16, INK2, anchor="ma")
    zoom(d, V, zx, zy, zc, t, shown, lab0[z], flag0[z], z, y, x)
    para(d, (zx, zy + 3 * zc + 52), "Each circle is one pair term: 0 if that pair of labels occurs in the solved "
                                    "designs, 1 if it never does. Grey cells are the neighbours' current labels.", 16,
         3 * zc, INK2)
    # the chart
    if stage >= 1:
        p = cost_chart(d, V, (1010, 212, 1870, 730), V["L"][t, ch], temp, stage >= 2, new_l, old)
    # the four steps
    d.line([(40, 790), (W - 40, 790)], fill=GRIDC, width=1)
    np = V["np"]
    cost = V["L"][t, ch][1:].astype(float)
    pr = np.exp(-(cost - cost.min()) / temp)
    pr /= pr.sum()
    low = int(cost.argmin()) + 1
    body = [f"One of the {n0} cells that break a rule, chosen at random.",
            "A label's cost here is its input cost plus its 8 pair costs with the neighbours' current labels.",
            (f"The label drawn had a {pr[new_l - 1] * 100:.0f}% chance. One more never-seen term makes a label "
             f"{float(np.exp(1 / temp)):.1f} times less likely."),
            f"Only this cell and its 8 neighbours can change state. Cells that break a rule: {n0} before, {n1} after."]
    bw = (W - 80 - 3 * 20) / 4
    for i in range(4):
        bx = 40 + i * (bw + 20)
        on = i == stage
        if on:
            d.rounded_rectangle([bx, 812, bx + bw, 1022], radius=12, fill=CARD, outline=GRIDC, width=1)
        col = INK if i <= stage else MUTED
        T(d, (bx + 22, 828), f"{i + 1}", 30, BLUE if on else col, bold=True)
        T(d, (bx + 56, 834), STEPS[i], 21, col, bold=True)
        if i <= stage:
            para(d, (bx + 22, 878), body[i], 18, bw - 44, INK if on else INK2)
    note = None
    if stage >= 2 and new_l != low:
        note = (f"The cheapest label had a {pr[low - 1] * 100:.0f}% chance and was not drawn. The randomness is what "
                "lets the search leave a dead end.")
    elif stage >= 3 and n1 == 0:
        note = "After this move no cell breaks a rule."
    if note:
        T(d, (W - 40, 1040), note, 16, INK2, anchor="ra")
    return img


def stat(d, x, y, w, big, small, col=INK):
    T(d, (x, y), big, 56, col, bold=True)
    para(d, (x, y + 72), small, 19, w, INK2)


def scene_trust(V, stage):
    img, d = new()
    dv, fr = V["dev"], V["fresh"]
    header(d, KICKER, "Why the stop can be trusted",
           "A stop means: every cell's terms were seen in solved designs. With the counted costs, no recorded stop "
           "ended on a wrong board.")
    stat(d, 80, 180, 480, f"{dv['exact']:,} of {dv['n']:,}", "dev designs built exactly, over four world sizes from "
                                                           "5×5×3 to 10×10×6, scored once.", GREEN)
    if stage >= 1:
        stat(d, 700, 180, 480, f"{fr['wrong_stops']} wrong stops", f"in {fr['runs']} more runs of {fr['n_per_run']:,} fresh "
             f"designs each. Between {fr['exact_min']:,} and {fr['exact_max']:,} were built exactly.", GREEN)
    if stage >= 2:
        stat(d, 1320, 180, 500, "The misses", "A design with no labelling of energy 0 under the fit (proved by a "
             "separate check), or a valid labelling that needs more bricks than the world's step limit.", INK)
    d.line([(40, 470), (W - 40, 470)], fill=GRIDC, width=1)
    if stage >= 2:
        y = para(d, (80, 510), "Why it holds", 24, 1700, INK, bold=True)
        F = V["fit"]
        assert F["forbidden_pair_terms_flagged"] == F["forbidden_pair_terms"]
        assert F["forbidden_input_terms_flagged"] == F["forbidden_input_terms"]
        y = para(d, (80, y + 8), f"The count marks as costly every term that the world's rules forbid: all "
                                 f"{F['forbidden_pair_terms']:,} forbidden pair terms and all "
                                 f"{F['forbidden_input_terms']} forbidden input terms, checked against the rules "
                                 "after the fit. So a labelling with no costly term breaks no rule.", 21, 1700, INK)
        y = para(d, (80, y + 26), "What the count gets wrong", 24, 1700, INK, bold=True)
        big = V["fitsize_impossible"]["16000"]
        assert sum(big) == 0
        para(d, (80, y + 8), f"It also marks {F['allowed_pair_terms_flagged']} allowed pair terms as costly, because "
                             f"they never occurred in the {F['n_train']:,} solved designs. A design that needs one of "
                             "them cannot be finished: the search runs to its move limit and reports no answer, "
                             f"rather than a wrong one. When the costs were counted from 16,000 solved designs, no "
                             f"fresh design was left without an answer for this reason ({len(big)} recorded "
                             "checks).", 21, 1700, INK)
    footer(d, "All numbers are read from the recorded result files. Fresh designs are new draws that are no dev, final, "
              "tuning or training design.")
    return img


RULES = {"tau": ("Energy threshold", "the paper's Sudoku rule"),
         "terms": ("Threshold per cost table", "its graph-coloring rule"),
         "local": ("Margin at every cell", "its N-Queens rule"),
         "fixed": ("No better single move", "its maze rule"),
         "light": ("Threshold per cost table", "same rule, lightly trained fit"),
         "count": ("Counted costs", "this project's fit")}


RULE_TEXT = {
    "tau": "Stop when the board's energy is no higher than the highest energy of a solved training design.",
    "terms": "Stop when no cost term is above the highest value its table takes in a solved training design.",
    "local": "Stop when every cell prefers its own label by at least the smallest margin found in solved training designs.",
    "fixed": "Stop when no single-cell change lowers the energy.",
    "light": "The same rule as the second board, with a fit trained for fewer and smaller steps.",
    "count": "Every cost is 0 or 1, so the rule becomes: no term that was never seen in a solved design.",
}


def light_share(V):
    f = next(b for b in V["paper"]["boards"] if b["key"] == "light")["fresh"]
    assert f["wrong_stops"] == 0
    return f"{100 * f['stopped_and_exact'] / f['n']:.0f}%"


def scene_paper(V, stage):
    img, d = new()
    P = V["paper"]
    B = P["boards"]
    header(d, KICKER, "The paper's own fit, with each of its four stopping rules",
           f"Costs learned by pseudolikelihood, as in the paper ({PAPER}). {P['fits_tried']} fits "
           f"({P['fits_tried'] - P['fits_usable']} unusable) and {P['temperatures']} temperatures were tried on "
           f"{P['n_tune']} tuning designs.")
    n = len(B)
    cw = (W - 80) / n
    T(d, (40, 138), "THIS DESIGN: THE BOARD EACH SEARCH STOPPED ON", 15, MUTED, bold=True)
    for i, b in enumerate(B):
        if stage < 1 and i > 0:
            break
        cx = 40 + cw * (i + 0.5)
        name, sub = RULES[b["key"]]
        T(d, (cx, 166), name, 19, INK, bold=True, anchor="ma")
        T(d, (cx, 192), sub, 15, INK2, anchor="ma")
        iso = iso_at(V, cx, 236, cw - 40, 170)
        lab, flag = V["blabels"][i].reshape(V["shape"]), V["bflags"][i].reshape(V["shape"])
        iso_cells(d, V, iso, lab, flag)
        yy = 418
        T(d, (cx, yy), f"stopped after {b['moves']} moves", 16, INK2, anchor="ma")
        if b["exact"]:
            MC.result_line(d, cx, yy + 26, True, "built exactly", 17)
        else:
            k = b["cells_breaking_a_true_rule"]
            MC.result_line(d, cx, yy + 26, False, f"wrong: {k} cell{'s' if k != 1 else ''} break a rule", 17)
    d.line([(40, 492), (W - 40, 492)], fill=GRIDC, width=1)
    if stage == 1:
        T(d, (40, 510), "WHEN EACH SEARCH STOPS", 15, MUTED, bold=True)
        for i, b in enumerate(B):
            para(d, (40 + cw * i + 24, 548), RULE_TEXT[b["key"]], 17, cw - 48, INK)
        y = para(d, (60, 760), "The search is the paper's focused search: it repairs cells that the costs mark as likely "
                               "wrong. With counted costs that is exactly the search shown before.", 21, 1780, INK)
        para(d, (60, y + 10), "The first four boards use the setting that finished the most tuning designs for that "
                              "rule. The fifth is a setting found afterwards.", 19, 1780, INK2)
    if stage >= 2:
        T(d, (40, 510), f"{P['n_fresh']:,} FRESH DESIGNS, FOUR WORLD SIZES: HOW EACH SEARCH ENDED", 15, MUTED, bold=True)
        x0, x1, y0, rh = 560, 1330, 560, 62
        cols = ((1440, "stopped, built exactly", GREEN, "stopped_and_exact"), (1620, "stopped on a wrong board", RED, "wrong_stops"),
                (1800, "not stopped", GREY, "not_stopped"))
        for cx, name, col, _ in cols:
            swatch(d, cx - 70, 524, col, 14)
            for j, line in enumerate(wrap(name, 14, 110)):
                T(d, (cx - 50, 520 + j * 17), line, 14, INK2)
        for i, b in enumerate(B):
            if stage < 3 and b["key"] in ("light", "count"):
                continue
            yy = y0 + i * rh + (14 if b["key"] in ("light", "count") else 0)
            name, sub = RULES[b["key"]]
            T(d, (60, yy + 4), name, 20, INK, bold=True)
            T(d, (60, yy + 30), sub, 15, INK2)
            f = b["fresh"]
            x = x0
            for _, _, col, key in cols:
                w = (x1 - x0) * f[key] / f["n"]
                if w >= 1:
                    d.rounded_rectangle([x, yy + 6, x + max(2, w - 2), yy + 40], radius=4, fill=col)
                x += w
            for cx, _, _, key in cols:
                T(d, (cx, yy + 12), f"{f[key]:,}", 20, INK, bold=key == "wrong_stops", anchor="ra")
            if b["key"] == "light":
                d.line([(40, yy - 6), (W - 40, yy - 6)], fill=GRIDC, width=1)
    if stage >= 2:
        lc = P["light_control"]
        msg = ("Each of the first four rows is the setting that finished the most tuning designs for that rule, scored "
               "once. Three rules stop on wrong boards most of the time.")
        if stage >= 3:
            msg = ("The lightly trained row was picked after the first four were read, then scored once on "
                   f"{lc['n']:,} new designs (counted costs there: {lc['stopped_and_exact']:,} built, "
                   f"{lc['wrong_stops']} wrong stops). It made no wrong stop and finished {light_share(V)} of them.")
        para(d, (60, 962), msg, 19, 1800, INK)
    footer(d, "Cells are marked with the world's true rules, for display only. All six use the same search code. A stop "
              "counts only if it also holds when rechecked in 64-bit arithmetic.")
    return img


def scene_gpu(V, frac):
    """frac in [0, 1]: the 512 recorded searches of the 10x10x6 dev designs, replayed by move count (log scale)."""
    import math
    img, d = new()
    dv, sp = V["dev"], V["speed"]
    mv = dv["w_moves"]
    header(d, KICKER, "On the GPU, every chain is one thread",
           "A custom Metal kernel runs the same moves as the reference search, with the same random numbers. Both "
           "are tested to agree move for move.")
    top = max(mv)
    cut = math.exp(frac * math.log(top + 1)) - 1 if frac < 1 else top
    done = sum(1 for m_ in mv if m_ <= cut)
    cols, pitch, x0, y0 = 32, 26, 60, 230
    T(d, (x0, y0 - 64), f"The {dv['w']['n']} dev designs of the 10×10×6 world, 4 chains each", 20, INK, bold=True)
    T(d, (x0, y0 - 36), "one square per design, filled once its first chain has stopped (recorded run)", 16, INK2)
    for i, m_ in enumerate(mv):
        bx, by = x0 + (i % cols) * pitch, y0 + (i // cols) * pitch
        if m_ <= cut:
            d.rounded_rectangle([bx, by, bx + pitch - 5, by + pitch - 5], radius=3, fill=GREEN)
        else:
            d.rounded_rectangle([bx, by, bx + pitch - 5, by + pitch - 5], radius=3, outline=GREY, width=2)
    yb = y0 + 16 * pitch + 20
    T(d, (x0, yb), f"after {int(cut):,} moves: {done} of {len(mv)} designs finished", 22, INK, bold=True)
    bx0, bx1 = x0, x0 + cols * pitch - 5
    d.rounded_rectangle([bx0, yb + 44, bx1, yb + 56], radius=6, fill=PLATE)
    d.rounded_rectangle([bx0, yb + 44, bx0 + max(12, (bx1 - bx0) * frac), yb + 56], radius=6, fill=GREEN)
    for m_ in (1, 10, 100, 1000):
        xx = bx0 + (bx1 - bx0) * math.log(m_ + 1) / math.log(top + 1)
        d.line([(xx, yb + 58), (xx, yb + 66)], fill=MUTED, width=2)
        T(d, (xx, yb + 68), f"{m_:,}", 14, MUTED, anchor="ma")
    T(d, (bx1, yb + 68), f"{top:,} moves", 14, MUTED, anchor="ra")
    x1 = 1000
    y = para(d, (x1, 170), "How the kernel is laid out", 24, 840, INK, bold=True)
    for line in ("One GPU thread owns one chain: its labels, and a flag per cell for 'breaks a rule'. No two threads "
                 "write the same memory, so nothing is locked.",
                 "One launch makes up to 256 moves of every chain that is still running. Between launches the chains "
                 "of a finished design are cancelled.",
                 "A move reads the 8 neighbours, adds up 25 costs, draws a label and rechecks 9 cells. Its cost does "
                 "not grow with the size of the design, apart from finding the cell to repair.",
                 "Random numbers come from a hash of (seed, chain, move), so a run can be repeated exactly, on the "
                 "CPU or the GPU."):
        d.ellipse([x1, y + 22, x1 + 8, y + 30], fill=INK2)
        y = para(d, (x1 + 24, y + 12), line, 19, 820, INK)
    if frac >= 1:
        d.line([(40, 790), (W - 40, 790)], fill=GRIDC, width=1)
        m16 = next(r for r in sp["metal"] if r["chains"] == 16384)
        stat(d, 80, 815, 520, f"{dv['w']['gpu_search_seconds']:.2f} s", f"of GPU search for all {dv['w']['n']} designs "
             f"(median {dv['w_moves_median']:g} moves, longest {dv['w']['moves_max']:,}). Laptop GPU.")
        stat(d, 700, 815, 520, f"{m16['moves_per_second'] / 1e6:.0f} million", f"moves per second at {m16['chains']:,} "
             "chains, in a throughput test on chains that never stop.")
        stat(d, 1320, 815, 520, f"{sp['numpy']['moves_per_second'] / 1e6:.1f} million", "moves per second for the NumPy "
             f"reference on 4 CPU threads, at {sp['numpy']['chains']:,} chains.")
    footer(d, "Timings include waiting for each launch and reading the stop flags back. One machine, one run each: read "
              "them as orders of size.")
    return img


CLOSE = [
    ("What the search is", "A random label on every cell, then one repair at a time: pick a cell that breaks a rule, "
     "score the 25 labels against its 8 neighbours, draw one. Stop at energy 0. No network and no reward."),
    ("What was designed, not learned", "The 25 labels come from the world's rules. The costs are counted from solved "
     "designs whose bricks are known. That is a stronger signal than the reward a network learns from."),
    ("Where it differs from the paper", "The energy's form, the 3×3 window, the focused search and the idea of a stop "
     "fitted to solved examples follow the paper. The fit does not: here it is a count, not gradient descent."),
    ("The paper's own fit, tested here", "With pseudolikelihood costs, three of the paper's four stopping rules "
     "stopped on wrong boards. Its graph-coloring rule with a lightly trained fit made no wrong stop on {n} fresh "
     "designs and finished {share} of them. That setting was picked with hindsight."),
    ("Not tested", "The paper's other training methods, other label sets, an annealed temperature, and the paper's "
     "own code."),
]


def scene_close(V, n):
    img, d = new()
    header(d, KICKER, "What this shows, and what it doesn't")
    y = 150
    for i, (head, body) in enumerate(CLOSE[:n]):
        T(d, (80, y), head, 24, INK, bold=True)
        body = body.format(n=f"{V['paper']['light_control']['n']:,}", share=light_share(V))
        y = para(d, (80, y + 38), body, 21, 1700, INK2 if i < n - 1 else INK) + 26
    footer(d, f"Paper: {PAPER}. Code, plans and every result file are in the project's repository.")
    return img


# ---------------------------------------------------------------------------------------------- timeline
def frames(V):
    """(image, number of video frames) pairs."""
    moves, nb = V["moves"], V["build"]["n_bricks"]
    out = [(scene_task(V, 0), 3 * FPS), (scene_task(V, 1), 5 * FPS), (scene_task(V, 2), 6 * FPS)]
    out += [(scene_labels(V, 0), 3 * FPS), (scene_labels(V, 1), 6 * FPS), (scene_labels(V, 2), 9 * FPS)]
    out += [(scene_energy(V, 0), 5 * FPS), (scene_energy(V, 1), 8 * FPS), (scene_energy(V, 2), 9 * FPS)]
    out += [(scene_search(V, 0, "start"), 7 * FPS)]
    for t in (0, moves - 1):
        out += [(scene_move(V, t, 0), 3 * FPS), (scene_move(V, t, 1), 6 * FPS), (scene_move(V, t, 2), 6 * FPS),
                (scene_move(V, t, 3), 5 * FPS)]
        if t == 0:
            out += [(scene_search(V, 1, "run"), 3 * FPS)]
            out += [(scene_search(V, k, "run"), 3 if k < 20 else 2) for k in range(2, moves - 1)]
            out += [(scene_search(V, moves - 1, "run"), 2 * FPS)]
    out += [(scene_search(V, moves, "stop"), 5 * FPS)]
    out += [(scene_search(V, moves, "build", 0), 2 * FPS)]
    out += [(scene_search(V, moves, "build", b), 3) for b in range(1, nb)]
    out += [(scene_search(V, moves, "build", nb), 6 * FPS)]
    out += [(scene_trust(V, 0), 3 * FPS), (scene_trust(V, 1), 5 * FPS), (scene_trust(V, 2), 11 * FPS)]
    out += [(scene_paper(V, 0), 3 * FPS), (scene_paper(V, 1), 11 * FPS), (scene_paper(V, 2), 9 * FPS),
            (scene_paper(V, 3), 11 * FPS)]
    steps = 36
    out += [(scene_gpu(V, 0), 2 * FPS)]
    out += [(scene_gpu(V, k / steps), 2) for k in range(1, steps)]
    out += [(scene_gpu(V, 1), 12 * FPS)]
    out += [(scene_close(V, k), 4 * FPS + 6) for k in range(1, len(CLOSE))]
    out += [(scene_close(V, len(CLOSE)), 9 * FPS)]
    return out


def stills():
    V = load()
    STILLS.mkdir(exist_ok=True)
    for old in STILLS.glob("*.png"):
        old.unlink()
    moves, nb = V["moves"], V["build"]["n_bricks"]
    shots = {"01_task": scene_task(V, 2), "02_labels": scene_labels(V, 2), "03_energy": scene_energy(V, 2),
             "04_start": scene_search(V, 0, "start"),
             "05_first_move_score": scene_move(V, 0, 1), "05_first_move_draw": scene_move(V, 0, 2),
             "05_first_move_done": scene_move(V, 0, 3),
             "06_search_mid": scene_search(V, moves // 2, "run"),
             "05_last_move_draw": scene_move(V, moves - 1, 2), "05_last_move_done": scene_move(V, moves - 1, 3),
             "07_stop": scene_search(V, moves, "stop"), "07_build_mid": scene_search(V, moves, "build", nb // 2),
             "07_build_done": scene_search(V, moves, "build", nb), "08_trust": scene_trust(V, 2),
             "09_paper_boards": scene_paper(V, 1), "09_paper": scene_paper(V, 3), "10_gpu_mid": scene_gpu(V, 0.6),
             "10_gpu": scene_gpu(V, 1), "11_close": scene_close(V, len(CLOSE))}
    for name, img in shots.items():
        img.save(STILLS / f"{name}.png")
    MC.check_words()
    print("stills:", ", ".join(shots))


def video():
    V = load()
    fr = frames(V)
    MC.check_words()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", "-crf",
                           "18", "-movflags", "+faststart", str(OUT)], stdin=subprocess.PIPE)
    for img, times in fr:
        buf = img.tobytes()
        for _ in range(times):
            ff.stdin.write(buf)
    ff.stdin.close()
    assert ff.wait() == 0
    scene_search(V, V["moves"] // 2, "run").save(HERE / "search_cover.png")
    print(json.dumps({"video": str(OUT), "seconds": round(sum(t for _, t in fr) / FPS, 1), "distinct_frames": len(fr),
                      "strings_checked": len(MC.SHOWN)}))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("collect", "stills", "video"):
        raise SystemExit(__doc__)
    if cmd != "collect":
        sys.path.insert(0, str(ENERGY))
    {"collect": collect, "stills": stills, "video": video}[cmd]()
