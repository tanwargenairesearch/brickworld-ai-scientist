"""Read-only status of the Stage 2e campaign: is the runner alive, how far is it, when should it end?

It reads files and asks the OS for the runner's process. It writes nothing, starts nothing and stops nothing.
It isn't in the campaign's manifest and the runner never imports it.

usage (from anywhere):  .venv/bin/python -I arch_rsi/stage2/2e/status_2e.py [CAMPAIGN_DIR]
"""
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
C = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "arch_rsi/campaigns/stage2e"
APPROVAL = ROOT / "arch_rsi/stage2/2e/LAUNCH_APPROVED.md"
CONSOLE = ROOT / "arch_rsi/campaigns/stage2e_console.log"
RUNS_FIXED, RUNS_PER_PAIR = 20, 134  # the protocol's upper bounds: 4,308 runs for 32 pairs, 6,452 for 48
STALL_MIN = 45


def load(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


def usd(files):
    return sum(((load(f, {}) or {}).get("usage") or {}).get("notional_cost_usd") or 0
               for f in files if not f.name.endswith(".failed.json"))


def main():
    now = datetime.now(timezone.utc)
    print(f"Stage 2e status at {now:%Y-%m-%d %H:%M} UTC | campaign: {C}")
    if not (C / "progress.log").exists():
        print("VERDICT: NOT STARTED. There is no progress.log in the campaign folder.")
        return
    log = (C / "progress.log").read_text().splitlines()
    done = any("STAGE2E_DONE" in line for line in log)
    ps = subprocess.run(["pgrep", "-fl", "run_2e.py --preset 2e"], capture_output=True, text=True).stdout
    alive = any("python" in line for line in ps.splitlines())
    stops = [n for n in ("INTEGRITY_STOP", "CEILING_STOP.json") if (C / n).exists()]

    results = list(C.glob("**/result.json"))
    newest = max([f.stat().st_mtime for f in results] + [(C / "progress.log").stat().st_mtime])
    quiet_min = (time.time() - newest) / 60

    entries = load(C / "confirm/summary.json", []) or []
    by_pair = {}
    for e in entries:
        if e.get("kind") == "search":
            by_pair.setdefault(e.get("pair"), set()).add(e.get("arm") or e.get("branch") or e.get("label"))
    pairs_confirmed = sum(len(v) >= 2 for v in by_pair.values())
    started = sorted(p.name for p in (C / "pairs").glob("p*")) if (C / "pairs").exists() else []
    rounds = {}
    for line in log:
        m = re.search(r"pairs/(p\d+)/(shared|seen|hidden) round (\d+)", line)
        if m:
            key = (m.group(1), m.group(2))
            rounds[key] = max(rounds.get(key, 0), int(m.group(3)))
    full = sum(rounds.get((p, "seen"), 0) >= 8 and rounds.get((p, "hidden"), 0) >= 8 for p in started)

    ss = load(C / "second_stage.json")
    target = 48 if (ss and ss.get("extend")) else 32
    clock = (load(C / "clock.json", {}) or {}).get("elapsed_s", 0.0)
    text = APPROVAL.read_text() if APPROVAL.exists() else ""
    ceil = {k: (float(m.group(1)) if (m := re.search(rf"^{k}:\s*([0-9.]+)\s*$", text, re.M)) else None)
            for k in ("ceiling_hours", "ceiling_usd")}
    calls = list(C.glob("pairs/*/*/round_*.json"))
    reviews = list((C / "review").glob("*.json")) if (C / "review").exists() else []
    failed = len(list(C.glob("pairs/*/*/*.failed.json")))
    spent = usd(calls) + usd(reviews)

    print(f"runner process alive: {alive} | STAGE2E_DONE in the log: {done} | stop files: {stops or 'none'}")
    print(f"runner hours so far: {clock / 3600:.1f} of ceiling {ceil['ceiling_hours']} | "
          f"notional LLM use: about ${spent:.0f} of ceiling {ceil['ceiling_usd']}")
    print(f"pairs started: {len(started)} | pairs with all 8 rounds in both branches: {full} | "
          f"pairs with both picks confirmed: {pairs_confirmed} | pairs planned now: {target}")
    print(f"training results on disk: {len(results)} of at most {RUNS_FIXED + RUNS_PER_PAIR * target} | "
          f"model calls saved: {len(calls)} search, {len(reviews)} review | failed call files: {failed}")
    if ss:
        print(f"second-stage rule: spread {ss.get('rms')} against {ss.get('threshold')}, extend to 48 pairs: {ss.get('extend')}")
    else:
        print("second-stage rule: not reached yet (it is read after pair 32)")
    print(f"minutes since the last log line or training result: {quiet_min:.0f}")
    print("last log lines:")
    for line in log[-4:]:
        print("   ", line[:230])

    eta = None
    if alive and not done and clock > 600 and len(results) > RUNS_FIXED:
        rate = (len(results) - RUNS_FIXED) / clock  # runs per second, after the fixed start-up runs
        left = max(RUNS_FIXED + RUNS_PER_PAIR * target - len(results), 0)
        eta = now + timedelta(seconds=left / rate)
        print(f"projected end at this pace, for {target} pairs: about {eta:%Y-%m-%d %H:%M} UTC, or sooner "
              f"(the run count is an upper bound; a move to 48 pairs would add about half as long again)")

    if done:
        print("VERDICT: DONE. STAGE2E_DONE is in progress.log. Next, in the experiment session: review_2e.py, then analyze_2e.py.")
    elif stops:
        print(f"VERDICT: STOPPED by {', '.join(stops)}. It needs a person: read the stop file and the last log lines.")
    elif not alive:
        print("VERDICT: STOPPED. The runner isn't running and STAGE2E_DONE isn't in the log. It needs a person. Console tail:")
        if CONSOLE.exists():
            for line in CONSOLE.read_text(errors="replace").splitlines()[-15:]:
                print("   ", line[:230])
    elif quiet_min > STALL_MIN:
        print(f"VERDICT: STALLED? The runner is alive, but nothing has landed for {quiet_min:.0f} minutes. It needs a person to look.")
    else:
        print("VERDICT: RUNNING." + (f" Projected end about {eta:%Y-%m-%d %H:%M} UTC or sooner." if eta else ""))


if __name__ == "__main__":
    main()
