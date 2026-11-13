"""VISUALIZATION ONLY: one video that puts the AI scientist's two kinds of suggestion side by side, in the same
world and on the same dev designs:
  1. inside the frozen search: networks and training recipes for learning from reward (PPO);
  2. outside the search: a change of method, to an energy counted from solved designs (energy_check/).
Nothing here scores, selects or tunes anything.

Where the second suggestion came from (README.md, "Provenance"): a person showed an LLM coding agent a paper on
energy-based reasoning and asked whether it applied to this world. The agent read the world's code, proposed the
labels and the energy, wrote and tested them, and the person approved each step. It is not a product of the
automated search, whose rules fix the training code, and it is a different LLM setup from the search's proposer.

Scenes, in order:
  1. the problem and the two suggestions;
  2. suggestion 1 building one design: the median starting network and the median network of the picks;
  3. suggestion 2 building the same design: its search, move by move, then its build;
  4. eight designs, smallest to largest, built by all three;
  5. every dev design of two worlds: the share of training runs that build it, beside the energy;
  6. the networks' training budget: what more practice does (earlier stages' own results);
  7. what differs, as a table;
  8. what the comparison shows and what it doesn't.

Rules, fixed before any of these builds was looked at:
- Worlds: W (10x10x6) for the builds; W and xl (8x8x5) for scene 5, the two worlds with confirmed networks.
- Designs (scene 4): sort W's 512 dev designs by number of cells, ties by dev index; take the designs at ranks
  round(i * 511 / 7), i = 0..7. Scenes 2 and 3 use the third of them (i = 2).
- Networks replayed: among the starting network's fresh-seed runs, and among the confirmed runs of the 16 picks
  of the "sees its scores" arm (g0), the run with the lower-median dev score; ties by (search, seed).
- Scene 5 uses every confirmed run: each design's share of runs that build it exactly, read from each run's dev.json.
- Network builds are greedy replays of saved checkpoints and must agree with each run's own dev.json.
- The energy's search is a replay of the scored run (run_learned.py dev): the same batch, settings and random
  numbers. It must give the recorded moves and brick counts for all 512 designs, and build the eight designs
  exactly in the frozen world, or `collect` stops. It rescores nothing.
- Every measured number on screen is read from a results file by `collect` and saved to compare_data.json. Scene 6
  reads the retest at twice the budget (arch_rsi/gpu/retest), Stage G's recipe test at four times (arch_rsi/gpu/stageG)
  and Stage 2a's budget ladder (arch_rsi/stage2/2a). Fixed facts of the set-up (25 labels, the 5x5x3 training world,
  four world sizes) are written in the text.
- No frame may hold a word of BANNED (checked on every string drawn).
- The final splits are never read.

usage (project root):
  .venv/bin/python -I -B energy_check/video/make_compare_video.py collect   # needs an Apple GPU, for a few seconds
  .venv/bin/python -I -B energy_check/video/make_compare_video.py stills    # one PNG per scene, in video/stills/
  .venv/bin/python -I -B energy_check/video/make_compare_video.py video     # video/energy_vs_network.mp4
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ENERGY = HERE.parent
ROOT = ENERGY.parent
VIZ2B = ROOT / "arch_rsi" / "viz" / "stage2b"
CAMPS = {"xl": (ROOT / "arch_rsi" / "campaigns" / "stage1", "start"),
         "w": (ROOT / "arch_rsi" / "campaigns" / "stage2b", "start_g")}
DATA, TRACE, OUT = HERE / "compare_data.json", HERE / "search_trace.npz", HERE / "energy_vs_network.mp4"
BANNED = ("gate", "claude", "lego", "fable", "opus", "sonnet", "haiku", "anthropic", "openai", "deepmind",
          "google", "prefrontal", "cram")
GALLERY_N, DEEP = 8, 2
GALLERY_RULE = ("W's 512 dev designs sorted by number of cells (ties by dev index); the designs at ranks "
                "round(i * 511 / 7), i = 0..7. Scenes 2 and 3 use the third (i = 2).")
NETWORK_RULE = "the run with the lower-median dev score in its group; ties by (search, seed)"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------------------------- collect
def collect():
    import numpy as np
    import torch

    torch.set_num_threads(4)
    sys.path.insert(0, str(ENERGY))
    SB = module("search_build", VIZ2B / "search_build.py")          # registers W; Stage 1's replay and drawing
    R = SB.R
    import run_learned as RL

    BC, BE, MS, RC = RL.BC, RL.BE, RL.MS, RL.RC
    assert not (ROOT / "arch_rsi" / "stage2" / "FINAL_OPENED_W").exists()
    rec = json.loads((RL.RESULTS / "learned_dev.json").read_text())
    hand = json.loads((RL.RESULTS / "dev.json").read_text())
    erow = {r["world"]: r for r in rec["rows"]}
    worlds = RC.worlds()

    def exact_of(run_dir):
        return np.array(json.loads((run_dir / "dev.json").read_text())["per_design"]["exact"], dtype=np.int64)

    def runs_of(camp, start_arm):
        """(results, start runs, pick runs); a run is (dev score, search, seed, run dir), sorted by the first three."""
        res = json.loads((camp / "results.json").read_text())
        assert res["primary_family"]["status"] == "final"
        sha = json.loads((camp / "comparators" / start_arm / "candidate.check.json").read_text())["sha"]
        start = [(float(exact_of(d).mean()), 0, int(d.name[4:]), d) for d in sorted((camp / "confirm" / sha).glob("seed*"))]
        assert len(start) == res["per_arm"][start_arm]["seeds"]
        assert abs(np.mean([s[0] for s in start]) - res["per_arm"][start_arm]["mean"]) < 6e-5
        picks = []
        for s in res["per_arm"]["g0"]["searches"]:
            mine = [(float(exact_of(d).mean()), s["search"], int(d.name[4:]), d)
                    for d in sorted((camp / "confirm" / s["sha"]).glob("seed*"))]
            assert len(mine) == len(s["per_seed"]) and abs(np.mean([m[0] for m in mine]) - s["primary_score"]) < 6e-5
            picks += mine
        assert abs(np.mean([p[0] for p in picks]) - res["per_arm"]["g0"]["mean"]) < 6e-5
        return res, sorted(start, key=lambda t: t[:3]), sorted(picks, key=lambda t: t[:3])

    # ---- scene 5: every dev design of the two worlds with confirmed networks
    out_worlds, kept = {}, {}
    for name, (camp, start_arm) in CAMPS.items():
        spec, dev_file, _ = worlds[name]
        designs = torch.load(dev_file)
        cells = designs.flatten(1).sum(1).numpy()
        res, start, picks = runs_of(camp, start_arm)
        s_share = np.stack([exact_of(t[3]) for t in start]).mean(0)
        p_share = np.stack([exact_of(t[3]) for t in picks]).mean(0)
        energy = np.array(erow[name]["per_design"]["exact"], dtype=np.int64)
        order = np.lexsort((np.arange(len(cells)), cells))                       # by cells, then dev index
        quarters = []
        for q in np.array_split(order, 4):
            quarters.append({"n": len(q), "cells_min": int(cells[q].min()), "cells_max": int(cells[q].max()),
                             "start": float(s_share[q].mean()), "picks": float(p_share[q].mean()),
                             "energy": float(energy[q].mean())})
        searches = res["per_arm"]["g0"]["searches"]
        out_worlds[name] = {
            "grid": [spec.X, spec.Y, spec.Z], "n": len(designs), "order": order.tolist(),
            "cells": cells.tolist(), "start_share": s_share.tolist(), "pick_share": p_share.tolist(),
            "energy": energy.tolist(), "start_runs": len(start), "pick_runs": len(picks),
            "searches": len(searches), "seeds_per_pick": len(searches[0]["per_seed"]),
            "start_mean": res["per_arm"][start_arm]["mean"], "picks_mean": res["per_arm"]["g0"]["mean"],
            "energy_exact": int(energy.sum()), "energy_not_stopped": erow[name]["not_stopped"],
            "energy_over_step_limit": erow[name]["stopped_over_step_limit"], "quarters": quarters,
            "start_built_by_no_run": int((s_share == 0).sum()), "picks_built_by_no_run": int((p_share == 0).sum()),
            "search_training_wall_hours": res["costs"]["g0"]["search_training_wall_hours"],
        }
        kept[name] = (spec, designs, cells, order, start, picks, s_share, p_share)

    # ---- the builds: W, eight designs by size, the two median networks, the energy
    spec, designs, cells, order, start, picks, s_share, p_share = kept["w"]
    spec = R.SPECS["w"]
    gal = [int(order[round(i * (len(order) - 1) / (GALLERY_N - 1))]) for i in range(GALLERY_N)]
    med = {"start": start[len(start) // 2 - 1], "pick": picks[len(picks) // 2 - 1]}

    def replay_network(run, di):
        score, search, seed, d = run
        src = (d / "run" / "candidate.py").read_text()
        ck = torch.load(d / "run" / "ckpt.pt", map_location="cpu", weights_only=False)
        assert ck["candidate_sha"] == R.sha(src), d
        bricks, end, occ = R.replay(SB.build_one(src, ck["state_dict"], designs[di], spec), designs[di], spec)
        exact = bool((occ == designs[di]).all())
        assert exact == bool(exact_of(d)[di]), f"{d}: the replay of design {di} disagrees with its dev.json"
        return {"bricks": [{"cells": [list(c) for c in b["cells"]], "wrong": b["wrong"]} for b in bricks], "end": end,
                "exact": exact, "missing": int((designs[di] & ~occ).sum()), "extra": int((occ & ~designs[di]).sum())}

    # the energy: a replay of the scored run on W, checked against its records for all 512 designs
    s = json.loads(RL.SETTINGS.read_text())["setting"]
    assert s == rec["setting"]
    tabs, fit = RL.fitted_tables(s["n_train"])
    for k in ("filled_cells", "parameters", "nonzero_parameters", "forbidden_pair_terms_flagged",
              "allowed_pair_terms_flagged", "forbidden_input_terms_flagged"):
        assert fit[k] == rec["fit"][k], k
    n, chains, dims = len(designs), s["chains"], (spec.Z, spec.Y, spec.X)
    tok = np.repeat(BC.tokens(designs).reshape(n, -1).numpy().astype(np.int32), chains, 0)
    RL.warm_up()
    y, found, _ = MS.search_gpu(tabs, tok, dims, s["max_moves"], s["temp"], s["seed"], chains=chains)
    f = found.reshape(n, chains)
    assert (f >= 0).any(1).all()
    first = np.where(f >= 0, f, 1 << 30).argmin(1)
    moves = f[np.arange(n), first]
    labels = torch.from_numpy(y.reshape(n, chains, *dims)[np.arange(n), first]).long()
    board = BC.decode(labels)
    pd = erow["w"]["per_design"]
    assert moves.tolist() == pd["moves"], "the replay's moves differ from the scored run's"
    assert board.flatten(1).sum(1).tolist() == pd["bricks"], "the replay's brick counts differ from the scored run's"
    acts = BE.bricks_to_actions(board, spec.done_action)

    gallery = []
    for di in gal:
        built, illegal = BE.build(acts[di:di + 1], designs[di:di + 1], spec)
        assert illegal == 0 and bool(RL.check(built.numpy(), designs[di:di + 1].numpy())["exact"][0]) == bool(pd["exact"][di])
        bricks, end, occ = R.replay(acts[di].tolist(), designs[di], spec)
        assert end == "pressed DONE" and bool((occ == designs[di]).all()) == bool(pd["exact"][di])
        zs = designs[di].nonzero()[:, 0]
        gallery.append({
            "index": di, "cells": int(cells[di]), "layers": int(zs.max()) + 1, "target": designs[di].int().tolist(),
            "start": replay_network(med["start"], di), "pick": replay_network(med["pick"], di),
            "energy": {"bricks": [{"cells": [list(c) for c in b["cells"]], "wrong": b["wrong"]} for b in bricks],
                       "exact": bool(pd["exact"][di]), "moves": int(moves[di]), "missing": int((designs[di] & ~occ).sum()),
                       "extra": int((occ & ~designs[di]).sum())},
            "start_runs_exact": int(round(s_share[di] * len(start))), "pick_runs_exact": int(round(p_share[di] * len(picks))),
        })

    # the search of the design of scenes 2 and 3, move by move (the same batch, stopped after k moves)
    di = gal[DEEP]
    rows = slice(di * chains, (di + 1) * chains)
    ys, founds = [], []
    for k in range(int(moves[di]) + 1):
        yk, fk, _ = MS.search_gpu(tabs, tok, dims, k, s["temp"], s["seed"], chains=chains)
        ys.append(yk[rows].copy())
        founds.append(fk[rows].copy())
    traj = np.stack(ys)
    chain = int(first[di])
    assert all((fk == -1).all() for fk in founds[:-1]) and founds[-1][chain] == moves[di]
    assert ((traj[1:] != traj[:-1]).sum(2) <= 1).all(), "a move changed more than one cell"
    assert (traj[-1, chain] == y[rows][chain]).all()
    nbr = BC.neighbours(*dims)
    flags = np.stack([BC.violations(yk, tok[rows], tabs, nbr) for yk in ys])
    assert not flags[-1, chain].any() and flags[:-1, chain].any(1).all()
    np.savez_compressed(TRACE, traj=traj.astype(np.int16), flags=flags, chain=chain, dims=np.array(dims))

    def net(run):
        res = json.loads((run[3] / "result.json").read_text())
        return {"dev": run[0], "search": run[1], "seed": run[2], "params": res["params"],
                "train_steps": res["train_steps"], "run": str(run[3].relative_to(ROOT))}

    man = json.loads((CAMPS["w"][0] / "manifest.json").read_text())["config"]
    man1 = json.loads((CAMPS["xl"][0] / "manifest.json").read_text())["config"]
    assert man["steps"]["w"] == man1["steps"]["xl"]
    retest = json.loads((ROOT / "arch_rsi" / "gpu" / "retest" / "runs" / "results.json").read_text())
    assert retest["world"] == "xl" and retest["split"] == "dev"
    g2 = json.loads((ROOT / "arch_rsi" / "gpu" / "stageG" / "runs" / "decisions.json").read_text())["G2"]
    g2row = next(t for t in g2["table"] if t["lr"] == g2["lr"] and t["n_minibatches"] == g2["n_minibatches"])
    lad = json.loads((ROOT / "arch_rsi" / "stage2" / "2a" / "runs" / "decisions.json").read_text())
    rung = next(x for x in lad["ladder"] if x["budget"] == 2 * man["steps"]["w"])
    assert lad["chosen_budget"] == man["steps"]["w"] and retest["budget"] == 2 * man1["steps"]["xl"]
    assert g2["budget"] == 4 * man1["steps"]["xl"]
    walls = sorted(json.loads((r_[3] / "result.json").read_text())["train_wall_s"] for r_ in picks)
    # designs with no labelling of energy 0 under a fit (run_fitsize.py): per number of training designs, one count
    # per training sample, each over all four worlds
    fs_rows = json.loads((RL.RESULTS / "learned_fitsize.json").read_text())["rows"]
    fitsize = {"n_per_run": sum(r_["n"] for r_ in fs_rows if (r_["fit_seed"], r_["n_train"]) ==
                                (fs_rows[0]["fit_seed"], fs_rows[0]["n_train"])), "impossible": {}}
    for r_ in fs_rows:
        per = fitsize["impossible"].setdefault(str(r_["n_train"]), {})
        per[str(r_["fit_seed"])] = per.get(str(r_["fit_seed"]), 0) + r_["impossible"]
        assert r_["impossible"] == r_["not_stopped"] and r_["not_stopped_and_not_proved_impossible"] == 0
    fitsize["impossible"] = {k: sorted(v.values()) for k, v in fitsize["impossible"].items()}
    fresh_rec = json.loads((RL.RESULTS / "learned_fresh.json").read_text())
    fresh_runs = {}
    for r_ in fresh_rec["rows"]:
        a = fresh_runs.setdefault((r_["fit_seed"], r_["search_seed"]), [0, 0, 0])
        a[0] += r_["exact"]
        a[1] += r_["n"]
        a[2] += r_["stopped_but_not_exact"] - r_["stopped_over_step_limit"]
    base = json.loads((CAMPS["w"][0] / "results.json").read_text())["baseline"]["w"]["check"]["budget"]
    data = {
        "world": {"X": spec.X, "Y": spec.Y, "Z": spec.Z, "n_designs": n, "max_steps": spec.max_steps},
        "rules": {"gallery": GALLERY_RULE, "networks": NETWORK_RULE},
        "worlds": out_worlds,
        "gallery": gallery, "deep": DEEP,
        "networks": {"start": net(med["start"]), "pick": net(med["pick"]),
                     "pick_dev_min": picks[0][0], "pick_dev_max": picks[-1][0],
                     "experiments_per_search": man["R"] * man["K"], "budget_steps": base["allowed_steps"],
                     "train_wall_s": {"min": walls[0], "median": walls[len(walls) // 2], "max": walls[-1],
                                      "runs_at_once": man["workers"]}},
        "more_practice": {
            "budget": man["steps"]["w"],
            "xl_x2": {"budget": retest["budget"], "start": retest["per_arm"]["start"]["mean"],
                      "start_runs": retest["per_arm"]["start"]["seeds"], "picks": retest["per_arm"]["g0"]["mean"],
                      "pick_runs": retest["per_arm"]["g0"]["seeds"]},
            "xl_x4": {"budget": g2["budget"], "start": g2row["mean"], "start_runs": len(g2row["per_seed"])},
            "w_x2": {"budget": rung["budget"], "reading": rung["reading"], "start": rung["cells"]["start_g"]["mean"],
                     "start_runs": rung["cells"]["start_g"]["n"], "refs": [rung["cells"]["refA"]["mean"],
                                                                           rung["cells"]["refB"]["mean"]],
                     "ref_runs": rung["cells"]["refA"]["n"]},
        },
        "fitsize": fitsize,
        "fresh": {"n_per_run": next(iter(fresh_runs.values()))[1], "runs": len(fresh_runs),
                  "exact_min": min(v[0] for v in fresh_runs.values()), "exact_max": max(v[0] for v in fresh_runs.values()),
                  "stopped_on_a_wrong_board": sum(v[2] for v in fresh_runs.values())},
        "energy": {
            "setting": s, "fit": fit, "fit_seconds_scored_run": rec["fit"]["fit_seconds"], "chains": chains,
            "chain": chain, "moves": int(moves[di]),
            # the median is taken from the per-design records (the file's summary field holds the lower middle value)
            "worlds": [{**{k: r[k] for k in ("world", "grid", "n", "exact", "moves_max", "gpu_search_seconds",
                                             "not_stopped", "stopped_over_step_limit")},
                        "moves_median": float(np.median(np.array(r["per_design"]["moves"])[
                            np.array(r["per_design"]["stopped"], bool)]))}
                       for r in rec["rows"]],
            "hand_set": {"exact": sum(r["exact"] for r in hand["rows"]), "n": sum(r["n"] for r in hand["rows"])},
        },
    }
    DATA.write_text(json.dumps(data))
    print("gallery (dev index, cells, layers):", [(g["index"], g["cells"], g["layers"]) for g in gallery])
    for g in gallery:
        print(f"  design {g['index']:3d}  {g['cells']:2d} cells | start {'ok ' if g['start']['exact'] else 'no '}"
              f"({g['start_runs_exact']}/{len(start)} runs) | pick {'ok ' if g['pick']['exact'] else 'no '}"
              f"({g['pick_runs_exact']}/{len(picks)} runs) | energy {'ok ' if g['energy']['exact'] else 'no '}"
              f"{g['energy']['moves']} moves, {len(g['energy']['bricks'])} bricks")
    print("networks:", json.dumps(data["networks"]))
    print("more practice:", json.dumps(data["more_practice"]))
    print("fresh:", json.dumps(data["fresh"]))
    for name, w in out_worlds.items():
        print(name, {k: w[k] for k in ("start_runs", "pick_runs", "start_mean", "picks_mean", "energy_exact",
                                       "start_built_by_no_run", "picks_built_by_no_run")})
        for q in w["quarters"]:
            print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in q.items()})
    print("deep design", di, "moves", int(moves[di]), "flagged by move:", flags[:, chain].sum(1).tolist())


# ---------------------------------------------------------------------------------------------- drawing
W, H, FPS = 1920, 1080, 12
INK, INK2, MUTED, GRIDC, SURF = (11, 11, 11), (82, 81, 78), (137, 135, 129), (225, 224, 217), (252, 252, 251)
START_COL, PICK_COL, ENERGY_COL = (217, 130, 43), (42, 120, 214), (27, 175, 122)   # validated together (dataviz)
BAD_COL, EMPTY, OTHER, CARD = (208, 59, 59), (232, 231, 225), (163, 162, 155), (246, 246, 242)
GHOST = (243, 214, 210)        # a target cell left unfilled
SHOWN = []
R = None          # Stage 1's isometric drawing, loaded by load()


def T(d, xy, s, size, col=INK, bold=False, anchor="la"):
    SHOWN.append(s)
    d.text(xy, s, font=R.font(size, bold), fill=col, anchor=anchor)


def wrap(s, size, width, bold=False):
    f, lines, cur = R.font(size, bold), [], ""
    for w in s.split():
        t = (cur + " " + w).strip()
        if f.getlength(t) <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + [cur]


def para(d, xy, s, size, width, col=INK, bold=False, lead=1.38):
    """Wrapped text; returns the y below its last line."""
    x, y = xy
    for line in wrap(s, size, width, bold):
        T(d, (x, y), line, size, col, bold)
        y += round(size * lead)
    return y


def new():
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H), SURF)
    return img, ImageDraw.Draw(img)


def header(d, kicker, title, sub=None):
    T(d, (40, 24), kicker.upper(), 15, MUTED, bold=True)
    assert R.font(32, True).getlength(title) <= 1840, f"the title is too long: {title!r}"
    T(d, (40, 48), title, 32, INK, bold=True)
    if sub:
        assert len(wrap(sub, 18, 1840)) == 1, f"the subtitle needs two lines: {sub!r}"
        para(d, (40, 96), sub, 18, 1840, INK2)


def swatch(d, x, y, col, size=16):
    d.rounded_rectangle([x, y, x + size, y + size], radius=3, fill=col)


def pct(v, digits=1):
    return "100%" if v == 1 else f"{v * 100:.{digits}f}%"


def load():
    global R
    import numpy as np
    R = module("r1", ROOT / "arch_rsi" / "viz" / "stage1" / "render_learning_stage1.py")
    V = json.loads(DATA.read_text())
    for g in V["gallery"]:
        g["target"] = np.array(g["target"], dtype=bool)
        for k in ("start", "pick", "energy"):
            g[k]["bricks"] = [{"cells": [tuple(c) for c in b["cells"]], "wrong": b["wrong"]} for b in g[k]["bricks"]]
    tr = np.load(TRACE)
    V["traj"], V["flags"], V["chain"] = tr["traj"], tr["flags"], int(tr["chain"])
    V["spec"] = SimpleNamespace(**{k: V["world"][k] for k in "XYZ"})
    V["d"] = V["gallery"][V["deep"]]
    assert V["traj"].shape[0] == V["energy"]["moves"] + 1 == V["d"]["energy"]["moves"] + 1
    return V


def iso_in(V, cx, top, width, layers, height=None):
    """An isometric view centred on cx whose tallest allowed point (`layers` high) is at y = top."""
    sp = V["spec"]
    sc = width / ((sp.X + sp.Y) * 0.866)
    if height:
        sc = min(sc, height / ((sp.X + sp.Y) * 0.5 + layers))
    return R.Iso(sp, sc, cx, top + layers * sc)


def with_ghosts(bricks, target):
    """The placed bricks, plus one pale unit cell for every target cell left unfilled."""
    filled = {c for b in bricks for c in b["cells"]}
    ghosts = [{"cells": [(int(z), int(y), int(x))], "wrong": False, "ghost": True}
              for z, y, x in zip(*target.nonzero()) if (int(z), int(y), int(x)) not in filled]
    return list(bricks) + ghosts


def draw_target(d, iso, target):
    iso.ground(d)
    iso.bricks(d, R.target_bricks(target), lambda b: R.TARGET_COL)


def draw_build(d, iso, target, bricks, col, ghosts=False):
    iso.ground(d, footprint=target.any(0))
    bl = with_ghosts(bricks, target) if ghosts else bricks
    iso.bricks(d, bl, lambda b: GHOST if b.get("ghost") else (BAD_COL if b["wrong"] else col))


def col_head(d, cx, head, sub):
    T(d, (cx, 138), head, 22, INK, bold=True, anchor="ma")
    T(d, (cx, 170), sub, 16, INK2, anchor="ma")


BIG_TOP, BIG_W, BIG_H = 214, 600, 500


def big_iso(V, cx):
    return iso_in(V, cx, BIG_TOP, BIG_W, V["d"]["layers"], BIG_H)


def big_bottom(V):
    iso = big_iso(V, 0)
    return iso.oy + (V["spec"].X + V["spec"].Y) * iso.b


def result_line(d, cx, y, ok, msg, size=18):
    w = R.font(size).getlength(msg)
    R.badge(d, cx - w / 2 - 16, y + 11, ok)
    T(d, (cx - w / 2 + 4, y), msg, size, INK)


def describe(r):
    pl = lambda n: f"{n} cell{'' if n == 1 else 's'}"  # noqa: E731
    if r["exact"]:
        return f"built exactly, {len(r['bricks'])} bricks"
    return f"not exact: {pl(r['missing'])} missing, {pl(r['extra'])} extra"


def col_target(d, V, cx):
    g = V["d"]
    col_head(d, cx, "Target design", "one of the dev designs")
    draw_target(d, big_iso(V, cx), g["target"])
    T(d, (cx, big_bottom(V) + 16), f"{g['cells']} cells in {g['layers']} layers", 17, INK2, anchor="ma")


def col_build(d, V, cx, head, sub, run, nb, col, done, extra=None):
    g = V["d"]
    col_head(d, cx, head, sub)
    draw_build(d, big_iso(V, cx), g["target"], run["bricks"][:nb], col, ghosts=done)
    y = big_bottom(V) + 16
    T(d, (cx, y), f"bricks placed: {min(nb, len(run['bricks']))}", 17, INK2, anchor="ma")
    if done:
        result_line(d, cx, y + 34, run["exact"], describe(run))
        if extra:
            T(d, (cx, y + 68), extra, 16, INK2, anchor="ma")


def draw_cells(d, iso, cells, group, colour, crossed):
    """Unit cells with one colour each. Outlines follow `group` (the brick a cell's label claims), and a cell in
    `crossed` gets a white X on its top face, so that the colour never carries the meaning alone.
    Painter's order by x + y + z, as in Stage 1's Iso.bricks."""
    occ, p = set(cells), iso.p
    ew = max(1, round(iso.s / 16))
    for (z, y, x) in sorted(cells, key=lambda c: (c[0] + c[1] + c[2], c[0])):
        col = colour[(z, y, x)]
        top, left, right, edge = R.mix(col, (255, 255, 255), 0.28), col, R.shade(col, 0.78), R.shade(col, 0.55)
        g = group[(z, y, x)]

        def same(cx, cy, cz):
            return group.get((cz, cy, cx)) == g

        zb, zt = z, z + 1
        show_top, show_r, show_l = (z + 1, y, x) not in occ, (z, y, x + 1) not in occ, (z, y + 1, x) not in occ
        if show_r:
            d.polygon([p(x + 1, y, zt), p(x + 1, y + 1, zt), p(x + 1, y + 1, zb), p(x + 1, y, zb)], fill=right)
        if show_l:
            d.polygon([p(x, y + 1, zt), p(x + 1, y + 1, zt), p(x + 1, y + 1, zb), p(x, y + 1, zb)], fill=left)
        if show_top:
            d.polygon([p(x, y, zt), p(x + 1, y, zt), p(x + 1, y + 1, zt), p(x, y + 1, zt)], fill=top)
            if not same(x, y - 1, z):
                d.line([p(x, y, zt), p(x + 1, y, zt)], fill=edge, width=ew)
            if not same(x - 1, y, z):
                d.line([p(x, y, zt), p(x, y + 1, zt)], fill=edge, width=ew)
            if not same(x + 1, y, z):
                d.line([p(x + 1, y, zt), p(x + 1, y + 1, zt)], fill=edge, width=ew)
            if not same(x, y + 1, z):
                d.line([p(x, y + 1, zt), p(x + 1, y + 1, zt)], fill=edge, width=ew)
            if (z, y, x) in crossed:
                d.line([p(x + .22, y + .5, zt), p(x + .78, y + .5, zt)], fill=(255, 255, 255), width=ew + 2)
                d.line([p(x + .5, y + .22, zt), p(x + .5, y + .78, zt)], fill=(255, 255, 255), width=ew + 2)
        if show_r:
            d.line([p(x + 1, y, zb), p(x + 1, y + 1, zb)], fill=edge, width=ew)
            if not same(x, y + 1, z):
                d.line([p(x + 1, y + 1, zt), p(x + 1, y + 1, zb)], fill=edge, width=ew)
            if not same(x, y - 1, z):
                d.line([p(x + 1, y, zt), p(x + 1, y, zb)], fill=edge, width=ew)
        if show_l:
            d.line([p(x, y + 1, zb), p(x + 1, y + 1, zb)], fill=edge, width=ew)
            if not same(x - 1, y, z):
                d.line([p(x, y + 1, zt), p(x, y + 1, zb)], fill=edge, width=ew)
            if not same(x + 1, y, z):
                d.line([p(x + 1, y + 1, zt), p(x + 1, y + 1, zb)], fill=edge, width=ew)


def search_state(V, k):
    """The chain shown after k moves: each filled cell's claimed brick (from its label) and whether it breaks a rule."""
    import brick_cram as BC
    sp = V["spec"]
    lab = V["traj"][k, V["chain"]].reshape(sp.Z, sp.Y, sp.X)
    flag = V["flags"][k, V["chain"]].reshape(sp.Z, sp.Y, sp.X)
    cells, group, colour, crossed = [], {}, {}, set()
    for z, y, x in zip(*V["d"]["target"].nonzero()):
        c = (int(z), int(y), int(x))
        t = int(BC.TYPE[lab[c]])
        dy, dx = BC.CELLS[t][int(BC.POS[lab[c]])]
        cells.append(c)
        group[c] = (c[0], t, c[1] - dy, c[2] - dx)
        colour[c] = BAD_COL if flag[c] else ENERGY_COL
        if flag[c]:
            crossed.add(c)
    return cells, group, colour, crossed


# ---------------------------------------------------------------------------------------------- scenes
def card(d, box, kicker, title, body, note, legend):
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, radius=14, fill=CARD, outline=GRIDC, width=1)
    T(d, (x0 + 32, y0 + 24), kicker.upper(), 15, MUTED, bold=True)
    T(d, (x0 + 32, y0 + 50), title, 28, INK, bold=True)
    y = para(d, (x0 + 32, y0 + 102), body, 21, x1 - x0 - 64, INK)
    y = para(d, (x0 + 32, y + 8), note, 18, x1 - x0 - 64, INK2)
    x = x0 + 32
    for col, name in legend:
        swatch(d, x, y + 14, col)
        T(d, (x + 24, y + 11), name, 17, INK)
        x += 24 + R.font(17).getlength(name) + 36
    assert y + 40 <= y1, (title, y, y1)


def scene_intro(V, stage):
    img, d = new()
    wd, N = V["world"], V["networks"]
    header(d, f"A brick-assembly world, {wd['X']}×{wd['Y']}×{wd['Z']}", "One problem, two suggestions from the AI scientist",
           "Build the target exactly, one brick at a time. Only an exact match counts.")
    g = V["d"]
    col_head(d, 400, "Target design", "one of the dev designs")
    draw_target(d, iso_in(V, 400, BIG_TOP, 620, g["layers"], BIG_H), g["target"])
    if stage >= 1:
        card(d, (800, 138, 1880, 448), "Suggestion 1 · inside the search", "Better networks for learning from reward",
             "A network places one brick per step and learns by trial and reward (PPO). The search lets the AI "
             "scientist (an LLM) change only the network and its training recipe.",
             f"{V['worlds']['w']['searches']} searches of {N['experiments_per_search']} experiments each in this world. "
             "The training code, the budget and the checker stay frozen, and the rules may not be written into the "
             "network.",
             [(START_COL, "the starting network"), (PICK_COL, "the AI scientist's networks")])
    if stage >= 2:
        card(d, (800, 468, 1880, 778), "Suggestion 2 · outside the search", "Change the method: an energy",
             "Label every filled cell with the piece of a brick it is. Count which neighbouring labels occur in solved "
             "designs. Then search for a labelling in which none of the others appears.",
             "Proposed by an LLM coding agent after a person showed it a paper on energy-based reasoning and this "
             "world's code. The person approved each step.",
             [(ENERGY_COL, "the AI scientist's energy")])
        d.line([(40, 812), (W - 40, 812)], fill=GRIDC, width=1)
        T(d, (40, 846), f"Both are tested on the same {wd['n_designs']} dev designs per world, by the same exact-match "
                        "checker.", 26, INK, bold=True)
        T(d, (40, 896), "They weren't suggested under the same rules, so this is not a like-for-like contest. "
                        "The held-out designs stay locked.", 20, INK2)
    return img


def scene_network(V, nb, done):
    img, d = new()
    N, g, ww = V["networks"], V["d"], V["worlds"]["w"]
    header(d, "Suggestion 1 · learning from reward", "A network places one brick per step and never takes one back",
           f"Each run trains by reward within one fixed, small budget: {N['budget_steps']:,} practice placements for "
           "the starting network, about as much compute for the others. No run is shown how a design is built.")
    col_target(d, V, 330)
    col_build(d, V, 960, "The starting network", "its median training run", g["start"], nb, START_COL, done,
              f"this design: built by {g['start_runs_exact']} of its {ww['start_runs']} training runs")
    col_build(d, V, 1590, "A network the AI scientist wrote", "the median run of its picks", g["pick"], nb, PICK_COL,
              done, f"this design: built by {g['pick_runs_exact']} of the picks' {ww['pick_runs']} training runs")
    if done:
        d.line([(40, 838), (W - 40, 838)], fill=GRIDC, width=1)
        swatch(d, 40, 868, GHOST, 20)
        T(d, (70, 866), "a target cell left unfilled", 18, INK)
        swatch(d, 330, 868, BAD_COL, 20)
        T(d, (360, 866), "a brick outside the target", 18, INK)
        swatch(d, 620, 868, (226, 230, 236), 20)
        T(d, (650, 866), "ground under the target", 18, INK)
        T(d, (40, 912), f"Over all {ww['n']} dev designs: the starting network builds {pct(ww['start_mean'])} exactly "
                        f"({ww['start_runs']} training runs), the AI scientist's picks {pct(ww['picks_mean'])} "
                        f"({ww['pick_runs']} runs).", 21, INK)
        T(d, (40, 956), f"The two runs shown score {pct(N['start']['dev'])} and {pct(N['pick']['dev'])}: the median "
                        "run of each group. The design is the third of eight chosen by size alone.", 18, INK2)
    return img


def chart(d, V, k, box):
    """Cells that break a rule, by search move, for the chains of this design, up to move k."""
    x0, y0, x1, y1 = box
    E, fl = V["energy"], V["flags"].sum(2)                          # [moves + 1, chains]
    moves, top = E["moves"], int(-(-fl.max() // 5) * 5)
    T(d, (x0 - 50, y0 - 46), "Cells that break a rule, by search move", 17, INK, bold=True)
    lx = x1 - 330
    d.line([(lx, y0 - 35), (lx + 26, y0 - 35)], fill=ENERGY_COL, width=3)
    T(d, (lx + 34, y0 - 46), "the chain shown", 15, INK2)
    d.line([(lx + 170, y0 - 35), (lx + 196, y0 - 35)], fill=OTHER, width=2)
    T(d, (lx + 204, y0 - 46), f"{fl.shape[1] - 1} other chains", 15, INK2)
    X = lambda m: x0 + (x1 - x0) * m / moves  # noqa: E731
    Y = lambda v: y1 - (y1 - y0) * v / top  # noqa: E731
    for v in range(0, top + 1, 5 if top <= 30 else 10):
        d.line([(x0, Y(v)), (x1, Y(v))], fill=GRIDC, width=1)
        T(d, (x0 - 12, Y(v)), str(v), 14, MUTED, anchor="rm")
    for m in range(0, moves + 1, 10 if moves <= 120 else 50):
        T(d, (X(m), y1 + 8), str(m), 14, MUTED, anchor="ma")
    T(d, (x1 + 12, y1 + 8), "moves", 14, MUTED)
    for c in [c for c in range(fl.shape[1]) if c != V["chain"]] + [V["chain"]]:
        col, wid = (ENERGY_COL, 3) if c == V["chain"] else (OTHER, 2)
        pts = []
        for m in range(k + 1):
            if m:
                pts.append((X(m), Y(fl[m - 1, c])))
            pts.append((X(m), Y(fl[m, c])))
        if len(pts) > 1:
            d.line(pts, fill=col, width=wid, joint="curve")
        if c == V["chain"]:
            cx, cy = X(k), Y(fl[k, c])
            d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=SURF)
            d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=col)


def scene_energy(V, k, nb, done):
    img, d = new()
    E, g = V["energy"], V["d"]
    fit, ww = E["fit"], next(r for r in E["worlds"] if r["world"] == "w")
    run = g["energy"]
    header(d, "Suggestion 2 · an energy", "The energy labels every cell at once, then repairs the cells that break a rule",
           f"Its {fit['parameters']:,} costs were counted from {fit['n_train']:,} solved designs of a smaller world "
           f"(5×5×3) in {E['fit_seconds_scored_run'] * 1000:.1f} ms. No reward and no network. The search runs on the "
           "laptop's GPU.")
    col_target(d, V, 330)
    col_head(d, 960, "The search", "one cell is relabelled per move")
    iso = big_iso(V, 960)
    iso.ground(d, footprint=g["target"].any(0))
    cells, group, colour, crossed = search_state(V, k)
    draw_cells(d, iso, cells, group, colour, crossed)
    y = big_bottom(V) + 16
    hidden = sum(1 for c in crossed if (c[0] + 1, c[1], c[2]) in set(cells))
    T(d, (960, y), f"move {k} of {E['moves']}  ·  cells that break a rule: {len(crossed)}"
                   + (f" ({hidden} under another cell)" if hidden else ""), 17, INK2, anchor="ma")
    if k < E["moves"]:
        x = 960 - 205
        swatch(d, x, y + 38, BAD_COL)
        d.line([(x + 4, y + 42), (x + 12, y + 50)], fill=(255, 255, 255), width=2)
        d.line([(x + 12, y + 42), (x + 4, y + 50)], fill=(255, 255, 255), width=2)
        T(d, (x + 24, y + 35), "breaks a rule", 17, INK)
        swatch(d, x + 170, y + 38, ENERGY_COL)
        T(d, (x + 194, y + 35), "no rule broken here", 17, INK)
    else:
        result_line(d, 960, y + 34, True, "no cell breaks a rule: stop")
    col_build(d, V, 1590, "The build", "the labels read as bricks, lowest layer first", run, nb, ENERGY_COL, done)
    d.line([(40, 838), (W - 40, 838)], fill=GRIDC, width=1)
    chart(d, V, k, (100, 906, 820, 1016))
    if not done:
        yy = para(d, (960, 860), "Start: a random label on every cell. Each move picks one cell that breaks a rule and "
                                 "gives it a new label, usually one that fits its neighbours better.", 20, 900, INK)
        para(d, (960, yy + 10), "Outlines show the brick each cell claims to be part of. "
                                f"{E['chains']} chains search at once and the first to finish is used. Shaded ground "
                                "is the target's footprint.", 18, 900, INK2)
    else:
        yy = para(d, (960, 860), f"This design: {E['moves']} search moves, then {len(run['bricks'])} bricks. The energy "
                                 "doesn't try to use few bricks.", 20, 900, INK)
        para(d, (960, yy + 10), f"All {ww['n']} dev designs of this world: {ww['exact']} built exactly, in "
                                f"{ww['moves_median']:g} moves at the median and {ww['moves_max']:,} at most. The GPU "
                                f"searched all {ww['n']} in {ww['gpu_search_seconds']:.2f} s.", 20, 900, INK)
    return img


GAL_ROWS = [("target", "Target", None), ("start", "Starting network", START_COL),
            ("pick", "AI scientist's network", PICK_COL), ("energy", "AI scientist's energy", ENERGY_COL)]


def scene_gallery(V, ncols, done):
    img, d = new()
    G, ww, N = V["gallery"], V["worlds"]["w"], V["networks"]
    header(d, "Both suggestions · eight designs", "Eight designs, smallest to largest, built by all three",
           f"Chosen by size alone: the {ww['n']} dev designs sorted by number of cells, every 73rd one taken. "
           "The networks are the median training run of each group, at the search's budget.")
    x0, colw, top, rowh = 250, 204, 150, 196
    layers = max(g["layers"] for g in G)
    for r, (key, name, col) in enumerate(GAL_ROWS):
        y = top + 34 + r * rowh
        if col:
            swatch(d, 40, y + 62, col, 18)
        for i, line in enumerate(wrap(name, 19, 170, True)):
            T(d, (66 if col else 40, y + 58 + i * 26), line, 19, INK, bold=True)
    tally = {"start": 0, "pick": 0, "energy": 0}
    for c, g in enumerate(G[:ncols]):
        cx = x0 + c * colw + colw // 2
        T(d, (cx, top), f"{g['cells']} cells", 18, INK, bold=True, anchor="ma")
        for r, (key, name, col) in enumerate(GAL_ROWS):
            y = top + 34 + r * rowh
            iso = iso_in(V, cx, y, colw - 14, layers, rowh - 44)
            if key == "target":
                draw_target(d, iso, g["target"])
                continue
            draw_build(d, iso, g["target"], g[key]["bricks"], col, ghosts=True)
            tally[key] += g[key]["exact"]
            by = y + rowh - 30
            note = {"start": f"{g['start_runs_exact']} of {ww['start_runs']} runs",
                    "pick": f"{g['pick_runs_exact']} of {ww['pick_runs']} runs",
                    "energy": f"{g['energy']['moves']:,} moves"}[key]
            wn = R.font(15).getlength(note)
            R.badge(d, cx - wn / 2 - 12, by + 9, g[key]["exact"], r=9)
            T(d, (cx - wn / 2 + 4, by), note, 15, INK2)
    d.line([(40, 968), (W - 40, 968)], fill=GRIDC, width=1)
    if done:
        x = 40
        T(d, (x, 990), f"Built exactly, of these {len(G)}:", 21, INK, bold=True)
        x += R.font(21, True).getlength(f"Built exactly, of these {len(G)}:") + 28
        for key, name, col in GAL_ROWS[1:]:
            swatch(d, x, 994, col, 18)
            s = f"{tally[key]} · {name}"
            T(d, (x + 26, 990), s, 21, INK)
            x += 26 + R.font(21).getlength(s) + 40
        T(d, (40, 1030), "The tick or cross is for the build shown. Beside it: how many of that group's training "
                         "runs build the design exactly, or the energy's search moves. Pale cells were left unfilled.",
          17, INK2)
    else:
        T(d, (40, 990), "The tick or cross is for the build shown. Beside it: how many of that group's training "
                        "runs build the design exactly, or the energy's search moves. Pale cells were left unfilled.",
          17, INK2)
    return img


def ramp(col, v):
    """Sequential ramp of one hue: the empty-cell grey at 0, the full colour at 1."""
    return R.mix(EMPTY, col, 0.12 + 0.88 * v) if v > 0 else EMPTY


def scene_grids(V, frac, done):
    img, d = new()
    header(d, "Both suggestions · every dev design", "Every dev design of two worlds: how often is it built exactly?",
           "One square per design, smallest at the top left, largest at the bottom right. Darker: built by a larger "
           "share of training runs, at the search's budget.")
    cols, pitch, sq = 32, 15, 13
    xs = [330, 870, 1410]
    T(d, (xs[0], 138), "The starting network", 22, INK, bold=True)
    T(d, (xs[1], 138), "The AI scientist's networks", 22, INK, bold=True)
    T(d, (xs[2], 138), "The AI scientist's energy", 22, INK, bold=True)
    for i, col in enumerate((START_COL, PICK_COL, ENERGY_COL)):
        swatch(d, xs[i] - 28, 142, col, 18)
    for r, name in enumerate(("xl", "w")):
        ww = V["worlds"][name]
        y0 = 196 + r * 400
        gx, gy, gz = ww["grid"]
        T(d, (40, y0 + 4), f"{gx}×{gy}×{gz} world", 24, INK, bold=True)
        T(d, (40, y0 + 40), f"{ww['n']} dev designs", 17, INK2)
        T(d, (40, y0 + 66), f"{min(ww['cells'])} to {max(ww['cells'])} cells each", 17, INK2)
        shown = round(frac * ww["n"])
        panels = [(ww["start_share"], START_COL, pct(ww["start_mean"]), f"of builds exact, over {ww['start_runs']} training runs"),
                  (ww["pick_share"], PICK_COL, pct(ww["picks_mean"]),
                   f"of builds exact, over {ww['pick_runs']} runs ({ww['searches']} picks × {ww['seeds_per_pick']} seeds)"),
                  (ww["energy"], ENERGY_COL, f"{ww['energy_exact']} of {ww['n']}", "designs built exactly, in one run")]
        for i, (vals, col, big, note) in enumerate(panels):
            for j, di in enumerate(ww["order"]):
                x, y = xs[i] + (j % cols) * pitch, y0 + (j // cols) * pitch
                d.rectangle([x, y, x + sq - 1, y + sq - 1], fill=ramp(col, vals[di]) if j < shown else EMPTY)
            if done:
                by = y0 + (ww["n"] // cols) * pitch + 12
                T(d, (xs[i], by), big, 34, INK, bold=True)
                T(d, (xs[i] + R.font(34, True).getlength(big) + 12, by + 14), note, 16, INK2)
                q = ww["quarters"][-1]
                v = [q["start"], q["picks"], q["energy"]][i]
                T(d, (xs[i], by + 48), f"largest quarter of designs: {pct(v)}", 17, INK2)
    if done:
        # the ramp's legend
        lx, ly = 40, 1010
        T(d, (lx, ly), "share of runs that build the design:", 16, INK2)
        x = lx + R.font(16).getlength("share of runs that build the design:") + 14
        T(d, (x, ly), "0%", 16, INK2)
        x += 34
        for i in range(11):
            d.rectangle([x + i * 20, ly + 2, x + i * 20 + 18, ly + 18], fill=ramp(PICK_COL, i / 10))
        T(d, (x + 228, ly), "100%", 16, INK2)
        F = V["fresh"]
        T(d, (x + 330, ly), "Energy: filled when its one run built the design. On "
                            f"{F['n_per_run'] // 4:,} fresh designs per world, {F['runs']} more runs built "
                            f"{pct(F['exact_min'] / F['n_per_run'], 2)} to {pct(F['exact_max'] / F['n_per_run'], 2)}.",
          16, INK2)
    return img


def scene_budget(V, stage):
    """What more practice does for the networks, from earlier stages' own results (8x8x5 dev designs)."""
    img, d = new()
    P, xl, w = V["more_practice"], V["worlds"]["xl"], V["worlds"]["w"]
    b = P["budget"]
    header(d, "Suggestion 1 · the training budget", "The networks' training budget was kept small on purpose",
           "A budget where networks already built about 90% left a search too little room to improve. "
           "Everything shown so far uses the smaller one.")
    # chart: exact-build rate against the budget, 8x8x5 world
    x0, y0, x1, y1 = 150, 250, 980, 800
    T(d, (60, 172), f"{xl['grid'][0]}×{xl['grid'][1]}×{xl['grid'][2]} world: dev designs built exactly, by practice "
                    "placements per training run", 20, INK, bold=True)
    Y = lambda v: y1 - (y1 - y0) * v  # noqa: E731
    xs = [x0 + 90, (x0 + x1) // 2, x1 - 90]
    for v in (0, .25, .5, .75, 1):
        d.line([(x0, Y(v)), (x1, Y(v))], fill=GRIDC, width=1)
        T(d, (x0 - 14, Y(v)), f"{v * 100:.0f}%", 16, MUTED, anchor="rm")
    for x, m in zip(xs, (1, 2, 4)):
        T(d, (x, y1 + 16), f"{b * m:,}", 18, INK, bold=True, anchor="ma")
        T(d, (x, y1 + 42), {1: "the search's budget", 2: "twice", 4: "four times"}[m], 16, INK2, anchor="ma")
    e = xl["energy_exact"] / xl["n"]
    for xa in range(x0, x1, 22):
        d.line([(xa, Y(e)), (min(xa + 12, x1), Y(e))], fill=ENERGY_COL, width=3)
    T(d, (x0 + 8, Y(e) - 30), f"the energy: {xl['energy_exact']} of {xl['n']}, with no practice", 18, INK)
    start = [xl["start_mean"], P["xl_x2"]["start"], P["xl_x4"]["start"]][:stage + 1]
    picks = [xl["picks_mean"], P["xl_x2"]["picks"]][:stage + 1]
    for vals, col, name, dy in ((start, START_COL, "the starting network", 30), (picks, PICK_COL, "the AI scientist's picks", -36)):
        pts = [(xs[i], Y(v)) for i, v in enumerate(vals)]
        if len(pts) > 1:
            d.line(pts, fill=col, width=4)
        for i, (px, py) in enumerate(pts):
            d.ellipse([px - 10, py - 10, px + 10, py + 10], fill=SURF)
            d.ellipse([px - 8, py - 8, px + 8, py + 8], fill=col)
            T(d, (px, py + (dy if dy > 0 else dy)), pct(vals[i]), 20, INK, bold=True, anchor="ma")
        T(d, (pts[0][0] + 26, pts[0][1] + (58 if dy > 0 else -64)), name, 17, INK2)
    # the reading, on the right
    x = 1060
    y = para(d, (x, 250), f"At the search's budget of {b:,} placements, the starting network builds "
                          f"{pct(xl['start_mean'])} and the picks {pct(xl['picks_mean'])}.", 22, 800, INK)
    if stage >= 1:
        y = para(d, (x, y + 22), f"At twice the budget: {pct(P['xl_x2']['start'])} and {pct(P['xl_x2']['picks'])}.",
                 22, 800, INK)
    if stage >= 2:
        y = para(d, (x, y + 22), f"At four times, the starting network alone builds {pct(P['xl_x4']['start'])} "
                                 f"({P['xl_x4']['start_runs']} runs). The picks weren't run there.", 22, 800, INK, bold=True)
        y = para(d, (x, y + 22), f"In the {w['grid'][0]}×{w['grid'][1]}×{w['grid'][2]} world, twice the budget was "
                                 f"rejected as too easy: two reference networks built {pct(P['w_x2']['refs'][0])} and "
                                 f"{pct(P['w_x2']['refs'][1])}.", 22, 800, INK)
        para(d, (x, y + 22), "So most of the gap to the energy comes from the training set-up, not the method. "
                             "With more practice and a better recipe, learning from reward gets close.", 22, 800, INK,
             bold=True)
        para(d, (60, 930), f"Runs: {xl['start_runs']} and {xl['pick_runs']} at the search's budget; "
                           f"{P['xl_x2']['start_runs']} and {P['xl_x2']['pick_runs']} at twice; "
                           f"{P['xl_x4']['start_runs']} at four times. The search's budget used the search's original "
                           "training recipe; the two larger budgets used a later, better one, so the rise is not from "
                           "practice alone.", 17, 1800, INK2)
    return img


def table_rows(V):
    N, E = V["networks"], V["energy"]
    fit, w, xl = E["fit"], V["worlds"]["w"], V["worlds"]["xl"]
    ew = {r["world"]: r for r in E["worlds"]}
    return [
        ("What it could change", "The network and its training recipe. Learning by reward (PPO), the budget and the "
                                 "checker are frozen, and the rules may not be written into the network.",
         "The whole method. It read a paper and the world's code, with tools and a person approving each step.", False),
        ("What the method is given", "A reward per brick for cells placed inside the target, a bonus for an exact "
                                     "match, a mask of legal moves, and maps of missing cells and heights. No solved "
                                     "designs.",
         f"{fit['n_train']:,} solved designs of the 5×5×3 world. Its 25 labels, its 'rests on something' input and its "
         "bottom-up build order were designed from the rules, not learned.", False),
        ("What is learned, and how", f"The network's weights, by trial and error. One training run took "
                                     f"{N['train_wall_s']['min']:.0f} to {N['train_wall_s']['max']:.0f} s "
                                     f"({N['train_wall_s']['runs_at_once']} at once); one search, about "
                                     f"{w['search_training_wall_hours'] / w['searches'] * 60:.0f} minutes of training.",
         f"{fit['parameters']:,} costs, by one pass of counting: {E['fit_seconds_scored_run'] * 1000:.1f} ms, after "
         "the solved designs are made. Designing it took a working session, which isn't timed.", False),
        ("How it builds", "One pass of the network per brick. No search, and no brick is taken back.",
         f"A search over all cells: {ew['w']['moves_median']:g} moves at the median in the largest world, "
         f"{ew['w']['moves_max']:,} at most. Then the bricks are placed.", False),
        (f"Built exactly, {xl['grid'][0]}×{xl['grid'][1]}×{xl['grid'][2]}, at the search's budget",
         f"{pct(xl['start_mean'])} start, {pct(xl['picks_mean'])} AI scientist's picks",
         f"{xl['energy_exact']} of {xl['n']}", True),
        (f"Built exactly, {w['grid'][0]}×{w['grid'][1]}×{w['grid'][2]}, at the search's budget",
         f"{pct(w['start_mean'])} start, {pct(w['picks_mean'])} AI scientist's picks",
         f"{w['energy_exact']} of {w['n']}", True),
    ]


def scene_table(V, nrows, foot):
    img, d = new()
    w, xl = V["worlds"]["w"], V["worlds"]["xl"]
    header(d, "Both suggestions · side by side", "What differs between the two suggestions")
    xa, xb, xc, wa = 40, 380, 1150, 700
    y = 124
    swatch(d, xb, y + 5, START_COL, 18)
    swatch(d, xb + 24, y + 5, PICK_COL, 18)
    T(d, (xb + 54, y - 2), "1 · Networks trained by reward", 26, INK, bold=True)
    swatch(d, xc, y + 5, ENERGY_COL, 18)
    T(d, (xc + 30, y - 2), "2 · An energy counted from solved designs", 26, INK, bold=True)
    y += 56
    d.line([(40, y), (W - 40, y)], fill=INK2, width=2)
    for label, a, b, big in table_rows(V)[:nrows]:
        size = 27 if big else 22
        y += 24
        yl = para(d, (xa, y), label, 22, 310, INK, bold=True)
        ya = para(d, (xb, y), a, size, wa, INK, bold=big)
        yb = para(d, (xc, y), b, size, wa + 30, INK, bold=big)
        y = max(yl, ya, yb) + 16
        d.line([(40, y), (W - 40, y)], fill=GRIDC, width=1)
    assert y < 980, y
    if foot:
        para(d, (40, 996), f"Networks: every build of every confirmed training run ({xl['start_runs']} runs of each "
                           f"starting network; {xl['pick_runs']} and {w['pick_runs']} runs of the picks). Energy: one "
                           "fit and one run of the search per world, with two settings chosen on 256 other designs "
                           "from each of the four worlds. All on dev designs.", 17, 1840, INK2)
    return img


CLOSE = [
    ("At the search's small training budget, the energy builds far more designs than the networks: {exact:,} of {n:,} "
     "dev designs across four world sizes, and {fmin} to {fmax} of fresh ones in five further runs.", True),
    ("That budget was kept small on purpose, so that a search had room to improve. With four times the practice and "
     "a better recipe, the starting network alone builds {x4} of the 8×8×5 designs. So this doesn't rank energies "
     "against reinforcement learning.", False),
    ("The two suggestions weren't made under the same rules. Inside the search, the rules may not be written into "
     "the network. The energy's labels were designed from the rules, it was given solved designs, and a person "
     "approved each step.", False),
    ("What it does show: once the rules are known, choosing the bricks is an easy puzzle. Rules written in by hand, "
     "with nothing learned, build {hexact:,} of {hn:,}. A high exact-build rate here isn't evidence of reasoning.",
     False),
]


def scene_close(V, n, foot):
    img, d = new()
    header(d, "Both suggestions · the reading", "What this comparison shows, and what it doesn't")
    E = V["energy"]
    F, P = V["fresh"], V["more_practice"]
    tot = {"exact": sum(r["exact"] for r in E["worlds"]), "n": sum(r["n"] for r in E["worlds"]),
           "hexact": E["hand_set"]["exact"], "hn": E["hand_set"]["n"],
           "fmin": pct(F["exact_min"] / F["n_per_run"], 2), "fmax": pct(F["exact_max"] / F["n_per_run"], 2),
           "x4": pct(P["xl_x4"]["start"])}
    y = 140
    for i, (line, bold) in enumerate(CLOSE[:n]):
        T(d, (60, y), str(i + 1), 30, MUTED, bold=True)
        y = para(d, (118, y), line.format(**tot), 29, 1720, INK, bold=bold, lead=1.4) + 34
    assert y < 940, y
    if foot:
        d.line([(40, 946), (W - 40, 946)], fill=GRIDC, width=1)
        S = V["fitsize"]
        few, many = sorted(S["impossible"], key=int)
        assert max(S["impossible"][many]) == 0
        para(d, (40, 962), "The energy's misses: tilings that need more bricks than the step limit, and designs that "
                           f"its count of {int(few):,} examples wrongly rules out ({min(S['impossible'][few])} to "
                           f"{max(S['impossible'][few])} of {S['n_per_run']:,} fresh designs over "
                           f"{len(S['impossible'][few])} samples of examples; none with {int(many):,} examples). "
                           "Limits: one laptop; the lattice world, with no physics; the held-out designs stay locked. "
                           "The energy follows the method of arXiv:2611.17309, but its fit here is counting, not that "
                           "paper's training objective.", 18, 1840, INK2)
    return img


# ---------------------------------------------------------------------------------------------- timeline
def frames(V):
    """(image, number of video frames) pairs."""
    E, g = V["energy"], V["d"]
    out = [(scene_intro(V, 0), 2 * FPS), (scene_intro(V, 1), 4 * FPS), (scene_intro(V, 2), 6 * FPS)]
    n = max(len(g["start"]["bricks"]), len(g["pick"]["bricks"]))
    out += [(scene_network(V, 0, False), 2 * FPS)]
    out += [(scene_network(V, k, False), 4) for k in range(1, n + 1)]
    out += [(scene_network(V, n, True), 7 * FPS)]
    moves, nb = E["moves"], len(g["energy"]["bricks"])
    per = 2 if moves <= 50 else 1
    step = max(1, -(-moves // 150))
    out += [(scene_energy(V, 0, 0, False), 3 * FPS)]
    out += [(scene_energy(V, k, 0, False), per) for k in range(step, moves, step)]
    out += [(scene_energy(V, moves, 0, False), FPS + 6)]
    out += [(scene_energy(V, moves, b, False), 2) for b in range(1, nb + 1)]
    out += [(scene_energy(V, moves, nb, True), 7 * FPS)]
    G = len(V["gallery"])
    out += [(scene_gallery(V, c, False), FPS + 3) for c in range(1, G)]
    out += [(scene_gallery(V, G, True), 7 * FPS)]
    steps = 24
    out += [(scene_grids(V, 0, False), FPS)]
    out += [(scene_grids(V, k / steps, False), 1) for k in range(1, steps + 1)]
    out += [(scene_grids(V, 1, True), 9 * FPS)]
    out += [(scene_budget(V, 0), 3 * FPS), (scene_budget(V, 1), 3 * FPS), (scene_budget(V, 2), 9 * FPS)]
    rows = len(table_rows(V))
    out += [(scene_table(V, r, False), 2 * FPS + 6) for r in range(1, rows)]
    out += [(scene_table(V, rows, True), 6 * FPS)]
    out += [(scene_close(V, k, False), 3 * FPS + 6) for k in range(1, len(CLOSE))]
    out += [(scene_close(V, len(CLOSE), True), 10 * FPS)]
    return out


def check_words():
    for s in SHOWN:
        low = s.lower()
        for b in BANNED:
            assert b not in low, f"banned word {b!r} on screen: {s!r}"
        assert "—" not in s and "–" not in s, s


def stills():
    V = load()
    out = HERE / "stills"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    E, g = V["energy"], V["d"]
    n = max(len(g["start"]["bricks"]), len(g["pick"]["bricks"]))
    rows, G = len(table_rows(V)), len(V["gallery"])
    shots = {"1_intro": scene_intro(V, 2), "2_network_mid": scene_network(V, n // 2, False),
             "2_network_done": scene_network(V, n, True), "3_energy_start": scene_energy(V, 0, 0, False),
             "3_energy_mid": scene_energy(V, E["moves"] // 2, 0, False),
             "3_energy_done": scene_energy(V, E["moves"], len(g["energy"]["bricks"]), True),
             "4_gallery_mid": scene_gallery(V, 3, False), "4_gallery_done": scene_gallery(V, G, True),
             "5_grids_mid": scene_grids(V, 0.5, False), "5_grids_done": scene_grids(V, 1, True),
             "6_budget": scene_budget(V, 2), "7_table": scene_table(V, rows, True),
             "8_close": scene_close(V, len(CLOSE), True)}
    for name, img in shots.items():
        img.save(out / f"{name}.png")
    check_words()
    print("stills:", ", ".join(shots))


def video():
    V = load()
    fr = frames(V)
    check_words()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", "-crf",
                           "18", "-movflags", "+faststart", str(OUT)], stdin=subprocess.PIPE)
    for img, times in fr:
        buf = img.tobytes()
        for _ in range(times):
            ff.stdin.write(buf)
    ff.stdin.close()
    assert ff.wait() == 0
    secs = sum(t for _, t in fr) / FPS
    scene_gallery(V, len(V["gallery"]), True).save(HERE / "cover.png")
    print(json.dumps({"video": str(OUT), "seconds": round(secs, 1), "distinct_frames": len(fr),
                      "strings_checked": len(SHOWN)}))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("collect", "stills", "video"):
        raise SystemExit(__doc__)
    if cmd != "collect":
        sys.path.insert(0, str(ENERGY))
    {"collect": collect, "stills": stills, "video": video}[cmd]()
