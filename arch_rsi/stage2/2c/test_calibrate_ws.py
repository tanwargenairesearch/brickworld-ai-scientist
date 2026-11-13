"""Tests for calibrate_ws.py with fake training runs: no GPU. Checks that 2a's ladder rules play out as in 2a for
seven cases (fallback chosen, too hard, a middle budget chosen; and Amendment 2's downward steps: one
chosen, too easy throughout, too hard on the way down, and not triggered when the fallback passes), that the noise step and the 2c projection
run, and that ws_budget.json is written only when a budget is chosen (to a scratch path here).
usage (project root): .venv/bin/python -I arch_rsi/stage2/2c/test_calibrate_ws.py SCRATCH_DIR
"""
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

import calibrate_ws as CW  # noqa: E402

SCRATCH = Path(sys.argv[1]).resolve() / "calib_test"


def fake(table):
    """table: budget -> {(net, recipe): mean}. Seeds vary by +-0.01 around the mean."""
    def run(cell):
        net, rk, seed, b = cell
        v = table[b][(net, rk)] + (0.01 if seed % 2 else -0.01)
        return {"net": net, "recipe": rk, "seed": seed, "budget": b, "status": "ok", "error": None, "steps": b,
                "flops_ratio": 1.0, "train_wall_s": 30.0, "dev_exact": v, "hidden": v, "learned": v >= 0.1, "wall_s": 40.0}
    return run


def cells(S, F, A, B):
    return {("start", "g"): S, ("start", "f"): F, ("refA", "g"): A, ("refB", "g"): B}


def case(name, table):
    out = SCRATCH / name
    if out.exists():
        shutil.rmtree(out)
    CW.CFG["full"] = dict(CW.CFG["full"], out=str(out))
    CW.BUDGET_PATH = out / "ws_budget.json"
    CW.run = fake(table)
    with contextlib.redirect_stdout(io.StringIO()):
        d = CW.main([])
    return d, out


if __name__ == "__main__":
    full = dict(CW.R2A.CFG["full"])
    assert (full["budgets"], full["fallback"], full["timeout"]) == ([262144, 524288, 1048576], 131072, 3600)
    assert (CW.R2A.EASY_S, CW.R2A.EASY_R, CW.R2A.HARD) == (0.50, 0.90, 0.20)
    easy, ok, hard = cells(0.6, 0.3, 0.95, 0.9), cells(0.3, 0.1, 0.7, 0.2), cells(0.05, 0.02, 0.1, 0.05)
    # 1. 262,144 too easy, the fallback passes: 131,072 chosen, noise step and budget file written
    d, out = case("fallback", {262144: easy, 131072: ok})
    assert d["status"] == "chosen" and d["chosen_budget"] == 131072 and [r["budget"] for r in d["ladder"]] == [262144, 131072]
    s = json.loads((out / "summary.json").read_text())["noise"]
    assert s["start_g"]["n"] == 20 and s["refA"]["n"] == 6 and s["start_f"]["n"] == 6
    p = s["projection_2c"]
    assert p["ws_runs"] == 1589 and p["w_runs"] == 2432 and p["eff_s_per_run_ws"] == 10.0
    assert abs(p["hours"] - (1589 * 10.0 + 2432 * p["eff_s_per_run_w_from_2a"]) / 3600) < 0.06
    assert json.loads((out / "ws_budget.json").read_text())["steps"] == 131072
    m = json.loads((out / "manifest.json").read_text())
    assert any(k.endswith("calib_test/fallback/start_frozen.py") for k in m["hashes"]) and m["world"] == "ws"
    # 3. every budget too hard: stop, no budget
    d, out = case("too_hard", {262144: hard, 524288: hard, 1048576: hard})
    assert d["status"] == "W_s too hard" and d["chosen_budget"] is None and not (out / "ws_budget.json").exists()
    # 4. 262k and 524k pass, 1M too easy: the largest passing budget is chosen
    d, out = case("middle", {262144: ok, 524288: ok, 1048576: easy})
    assert d["chosen_budget"] == 524288 and len(d["ladder"]) == 3
    # Amendment 2: the fallback too easy, then halving: 65,536 too easy, 32,768 passes -> chosen, 16,384 not run
    d, out = case("down_pass", {262144: easy, 131072: easy, 65536: easy, 32768: ok})
    assert d["status"] == "chosen" and d["chosen_budget"] == 32768
    assert [r["budget"] for r in d["ladder"]] == [262144, 131072, 65536, 32768]
    assert json.loads((out / "ws_budget.json").read_text())["steps"] == 32768
    # too easy all the way down: stop, no budget
    d, out = case("down_easy", {262144: easy, 131072: easy, 65536: easy, 32768: easy, 16384: easy})
    assert d["status"] == "W_s too easy" and d["chosen_budget"] is None and len(d["ladder"]) == 5
    assert not (out / "ws_budget.json").exists()
    # too easy at 131,072, too hard at 65,536: nothing passes, stop without going lower
    d, out = case("down_hard", {262144: easy, 131072: easy, 65536: hard})
    assert d["chosen_budget"] is None and d["status"].startswith("no budget passes") and len(d["ladder"]) == 3
    # the fallback passing never triggers the downward steps
    d, _ = case("fallback_again", {262144: easy, 131072: ok})
    assert [r["budget"] for r in d["ladder"]] == [262144, 131072]
    # the frozen-recipe candidate is the start plus 2b's frozen block, and "learned" uses W_s's hidden-cell designs
    src = (out / "start_frozen.py").read_text()
    assert CW.w2b.parse_recipe(src) == CW.w2b.FROZEN_AS_BLOCK and 0.7 < CW.H.mean() < 0.95
    print("all calibrate_ws tests passed")
