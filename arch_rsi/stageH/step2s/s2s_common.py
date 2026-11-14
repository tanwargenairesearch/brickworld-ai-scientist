"""Shared paths and settings for Step 2S (PROTOCOL_STEP2S.md). Step 2's settings are imported, not copied."""
import sys
from pathlib import Path

STEP2S = Path(__file__).resolve().parent
for _p in (str(STEP2S.parent / "step2"), str(STEP2S)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import s2common as C2  # noqa: E402  (Step 2's paths and settings, unchanged; it puts Stage H1 and the root on the path)

ROOT, STAGEH, STEP2, common = C2.ROOT, C2.STAGEH, C2.STEP2, C2.common
FOUND = C2.FOUND                                   # 38.3%, Step 2's threshold
TARGETS2S = STEP2S / "targets"
PROBE_DESIGNS = TARGETS2S / "probe.pt"
PROBE_SEED, N_PROBE_DESIGNS, MAX_PROBES, MAX_PLACEMENTS = 4747, 4, 3, 16
ARMS = ("sandbox", "scores", "told_sandbox")
SANDBOX_ARMS = ("sandbox", "told_sandbox")
STEP2_RESULTS = STEP2 / "campaigns" / "step2" / "results.json"
MIN_TOLD_FOUND, MAX_SCORES_FOUND, PER_WORLD_N = 9, 9, 12   # "which worlds run": told >= 9 of 12, scores < 9 of 12
SEARCHES = {2: {"sandbox": 12, "scores": 12, "told_sandbox": 6}, 1: {"sandbox": 24, "scores": 24, "told_sandbox": 12}}


def qualify(res: dict) -> dict:
    """Which worlds Step 2S runs in (PROTOCOL_STEP2S.md, "Which worlds run"), from Step 2's final full-run results."""
    if res.get("status") != "final" or res.get("preset") != "step2":
        raise SystemExit("Step 2S needs Step 2's final full-run results (status final, preset step2)")
    per = {}
    for w in C2.WORLDS:
        told = [r["score"] for r in res["rows"] if r["world"] == w and r["arm"] == "told"]
        sc = [r["score"] for r in res["rows"] if r["world"] == w and r["arm"] == "scores"]
        if len(told) != PER_WORLD_N or len(sc) != PER_WORLD_N:
            raise SystemExit(f"expected {PER_WORLD_N} told and scores-only searches in {w}, got {len(told)} and {len(sc)}")
        tf, sf = sum(s >= FOUND for s in told), sum(s >= FOUND for s in sc)
        per[w] = {"told_found": tf, "scores_found": sf, "n": PER_WORLD_N,
                  "implementable": tf >= MIN_TOLD_FOUND, "room_left": sf < MAX_SCORES_FOUND}
        per[w]["runs"] = per[w]["implementable"] and per[w]["room_left"]
    worlds = [w for w in C2.WORLDS if per[w]["runs"]]
    return {"per_world": per, "worlds": worlds, "searches_per_world": SEARCHES.get(len(worlds))}
