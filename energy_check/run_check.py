"""Tune and score the hand-set brick energy (brick_energy.py) on the brick-assembly worlds.

usage (project root):
  .venv/bin/python -I -B energy_check/run_check.py tune     # every setting of GRID on fresh tuning designs
  .venv/bin/python -I -B energy_check/run_check.py freeze   # pick one setting by RULE and write settings.json
  .venv/bin/python -I -B energy_check/run_check.py dev      # score each dev set once with settings.json

Protocol.
- Tuning designs: 256 per world from the frozen generator (seed TUNE_SEED), with every dev and final design rejected by
  hash, as the training stream does. No setting is chosen on a dev design.
- RULE: the setting that builds the most tuning designs exactly, summed over the worlds; ties go to the fewest search
  moves (mean over stopped designs, summed over the worlds).
- `dev` refuses to run without settings.json, and refuses to run twice: it writes results/dev.started before it
  scores anything and refuses when that file or results/dev.json exists. The guard is a file check and no more.
- The final splits (holdout/, arch_rsi/stage2/holdout/) are never read.

Every build is made by stepping the frozen `BrickWorld` and scored by the frozen checker `evaluate.check`.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import brick_energy as BE  # noqa: E402
from arch_rsi.worlds import SPECS  # noqa: E402
from brickworld.env import SCALES, StreamSource, design_hash, zobrist_codes  # noqa: E402
from evaluate import check  # noqa: E402

RESULTS = HERE / "results"
SETTINGS = HERE / "settings.json"
TUNE_SEED = 20261113
N_TUNE = 256
GRID = [{"mu": mu, "temp": temp, "chains": 4, "max_moves": 20000, "seed": 0}
        for mu in (0.0, 0.5) for temp in (0.25, 0.5, 1.0)]
RULE = "most tuning designs built exactly, summed over the worlds; ties: fewest mean search moves"


def worlds() -> dict:
    """name -> (spec, dev designs file, file of dev and final hashes). W's spec is read from its own module."""
    spec = importlib.util.spec_from_file_location("w_world", ROOT / "arch_rsi" / "stage2" / "w_world.py")
    ww = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ww)
    return {
        "small": (SCALES["small"], ROOT / "targets/small_dev.pt", ROOT / "targets/small_heldout_hashes.pt"),
        "large": (SCALES["large"], ROOT / "targets/large_dev.pt", ROOT / "targets/large_heldout_hashes.pt"),
        "xl": (SPECS["xl"], ROOT / "arch_rsi/targets/xl_dev.pt", ROOT / "arch_rsi/targets/xl_heldout_hashes.pt"),
        "w": (ww.W, ROOT / "arch_rsi/stage2/targets/w_dev.pt", ROOT / "arch_rsi/stage2/targets/w_heldout_hashes.pt"),
    }


def tuning_designs(spec, hashes_file) -> torch.Tensor:
    held = torch.load(hashes_file)
    designs = StreamSource(spec, TUNE_SEED, held, chunk=2048)(N_TUNE).clone()
    assert not bool(torch.isin(design_hash(designs, zobrist_codes(spec)), held).any())
    return designs


def score(spec, designs: torch.Tensor, s: dict, per_design: bool = False) -> dict:
    """Search every design, build each answer in the frozen world and score it with the frozen checker."""
    t0 = time.perf_counter()
    out = BE.search(designs, cap=spec.max_steps, mu=s["mu"], temp=s["temp"], chains=s["chains"],
                    max_moves=s["max_moves"], seed=s["seed"])
    seconds = time.perf_counter() - t0
    built, illegal = BE.build(BE.bricks_to_actions(out["board"], spec.done_action), designs, spec)
    exact = torch.from_numpy(check(built.numpy(), designs.numpy())["exact"])
    stopped = out["stopped"]
    moves = out["moves"][stopped].float()
    cells = designs.flatten(1).sum(1)
    res = {
        "world": spec.name, "grid": [spec.X, spec.Y, spec.Z], "n": len(designs),
        "exact": int(exact.sum()), "exact_rate": float(exact.float().mean()),
        "stopped": int(stopped.sum()),
        "stopped_but_not_exact": int((stopped & ~exact).sum()),
        "not_stopped": int((~stopped).sum()),
        "illegal_placements": illegal,
        "moves_mean": float(moves.mean()) if len(moves) else None,
        "moves_median": float(moves.quantile(0.5)) if len(moves) else None,      # the mean of the two middle values
        "moves_p90": float(moves.quantile(0.9)) if len(moves) else None,
        "moves_max": int(moves.max()) if len(moves) else None,
        "bricks_mean": float(out["bricks"][stopped].float().mean()) if len(moves) else None,
        "bricks_max": int(out["bricks"][stopped].max()) if len(moves) else None,
        "brick_cap": spec.max_steps,
        "designs_with_more_cells_than_cap": int((cells > spec.max_steps).sum()),
        "share_with_hidden_cell": float(BE_hidden(designs)),
        "search_seconds": round(seconds, 2),
    }
    if per_design:
        res["per_design"] = {"exact": exact.int().tolist(), "stopped": stopped.int().tolist(),
                             "moves": out["moves"].tolist(), "bricks": out["bricks"].tolist()}
    return res


def BE_hidden(designs: torch.Tensor) -> torch.Tensor:
    from brickworld.env import has_hidden_cells
    return has_hidden_cells(designs).float().mean()


def dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def cmd_tune(args):
    W = worlds()
    names = args.worlds or list(W)
    grid = GRID if args.setting is None else [GRID[args.setting]]
    rows = []
    for s in grid:
        for name in names:
            spec, _, hashes = W[name]
            r = score(spec, tuning_designs(spec, hashes), s)
            rows.append({"setting": s, **r})
            print(json.dumps({"mu": s["mu"], "temp": s["temp"], **{k: r[k] for k in (
                "world", "exact", "n", "not_stopped", "stopped_but_not_exact", "moves_median", "moves_max",
                "bricks_max", "brick_cap", "search_seconds")}}), flush=True)
    if args.worlds or args.setting is not None:
        print("partial run: nothing written")
        return
    dump({"tune_seed": TUNE_SEED, "n_tune": N_TUNE, "rule": RULE, "rows": rows}, RESULTS / "tune.json")


def cmd_freeze(args):
    rec = json.loads((RESULTS / "tune.json").read_text())
    table = []
    for s in GRID:
        rows = [r for r in rec["rows"] if r["setting"] == s]
        assert len(rows) == len(worlds())
        table.append((-sum(r["exact"] for r in rows), sum(r["moves_mean"] or float("inf") for r in rows), s))
    table.sort(key=lambda it: (it[0], it[1]))
    best = table[0]
    dump({"setting": best[2], "rule": rec["rule"], "tune_exact_total": -best[0], "tune_n_total": N_TUNE * len(worlds()),
          "tune_moves_mean_sum": best[1]}, SETTINGS)
    for neg, mv, s in table:
        print(f"mu {s['mu']:<4} temp {s['temp']:<5} exact {-neg}/{N_TUNE * len(worlds())}  mean moves (sum over worlds) {mv:.1f}")
    print("frozen:", json.dumps(best[2]))


def cmd_dev(args):
    if not SETTINGS.exists():
        raise SystemExit("refused: no settings.json (run tune, then freeze)")
    out, started = RESULTS / "dev.json", RESULTS / "dev.started"
    if out.exists() or started.exists():
        raise SystemExit("refused: results/dev.json or .started exists; each dev set is scored once")
    s = json.loads(SETTINGS.read_text())["setting"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    started.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")      # a run that began counts as the one run
    rows = []
    for name, (spec, dev, _) in worlds().items():
        r = score(spec, torch.load(dev), s, per_design=True)
        rows.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "per_design"}), flush=True)
    dump({"setting": s, "rows": rows}, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--worlds", nargs="*")
    t.add_argument("--setting", type=int)
    sub.add_parser("freeze")
    sub.add_parser("dev")
    args = ap.parse_args()
    torch.set_num_threads(4)
    {"tune": cmd_tune, "freeze": cmd_freeze, "dev": cmd_dev}[args.cmd](args)
