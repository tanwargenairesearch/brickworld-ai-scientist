"""Step 3 program helpers (PROTOCOL_STEP3.md, "The program and its header"). Pure functions: no training, no calls.

The header has three fields: rounds (2-6), candidates_per_round (2-6) and seeds_per_candidate (1-3), with
rounds x candidates x seeds <= 24, Step 2's trial runs per search. 2c's retest step and network/recipe split are
dropped. The facts-only lock is 2c's (prog2c.facts_only_problems) with this header's parser passed in; a test shows it
returns exactly what 2c's does on 2c's programs. The rewrite logs reuse prog2c.format_logs per search, add the search's
world as a neutral label ("world 1" to "world 7") and its per-round analyses (this step has no shuffled control).
"""
import random
import re
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
_2C = HERE.parents[1] / "stage2" / "2c"
if str(_2C) not in sys.path:
    sys.path.insert(0, str(_2C))
import prog2c as P2C  # noqa: E402  (Stage 2c's helpers, built and tested, never run)

MAX_TRIAL_RUNS = 24
BOUNDS = {"rounds": (2, 6), "candidates_per_round": (2, 6), "seeds_per_candidate": (1, 3)}
G0_HEADER = {"rounds": 4, "candidates_per_round": 3, "seeds_per_candidate": 2}
_HEADER_RE = P2C._HEADER_RE
_SWAP = threading.Lock()
strip_header = P2C.strip_header
sections = P2C.sections


def render_header(h: dict) -> str:
    return ("<!-- header\n"
            f"rounds: {h['rounds']}\n"
            f"candidates_per_round: {h['candidates_per_round']}\n"
            f"seeds_per_candidate: {h['seeds_per_candidate']}\n"
            "-->\n")


def trial_runs(h: dict) -> int:
    return h["rounds"] * h["candidates_per_round"] * h["seeds_per_candidate"]


def validate_header(h: dict) -> list:
    out = []
    for k, (lo, hi) in BOUNDS.items():
        if k not in h:
            out.append(f"missing {k}")
        elif not isinstance(h[k], int) or not lo <= h[k] <= hi:
            out.append(f"{k} must be {lo}-{hi}, got {h[k]}")
    extra = set(h) - set(BOUNDS)
    if extra:
        out.append(f"unknown header keys: {sorted(extra)}")
    if not out and trial_runs(h) > MAX_TRIAL_RUNS:
        out.append(f"rounds x candidates_per_round x seeds_per_candidate = {trial_runs(h)}, over {MAX_TRIAL_RUNS}")
    return out


def parse_program(text: str):
    """Returns (header, body). Raises ValueError with a message the rewrite can be shown if the header is bad."""
    m = _HEADER_RE.match(text)
    if not m:
        raise ValueError("the program must start with a <!-- header ... --> block")
    h = {}
    for line in m.group(1).splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"header line without a colon: {line!r}")
        k, v = (s.strip() for s in line.split(":", 1))
        if not re.fullmatch(r"-?\d+", v):
            raise ValueError(f"header value for {k} must be a whole number, got {v!r}")
        h[k] = int(v)
    problems = validate_header(h)
    if problems:
        raise ValueError("; ".join(problems))
    return h, text[m.end():]


def g0_program(g0r_text: str) -> str:
    """g0: g0r.md unchanged, with the header Step 2 effectively ran (4 rounds of 3 candidates, 2 seeds each)."""
    return render_header(G0_HEADER) + "\n" + g0r_text


def settings(h: dict) -> dict:
    return {"R": h["rounds"], "K": h["candidates_per_round"], "trial_seeds": list(range(h["seeds_per_candidate"]))}


def facts_only_problems(parent: str, child: str, max_words: int = 1000, parse=None) -> list:
    """2c's facts-only check, with this step's header parser. Implemented by calling 2c's function with its module's
    parse_program swapped for the call, so the logic is 2c's own code."""
    parse = parse or parse_program
    with _SWAP:                                   # rewrites run in threads; the swap is one call at a time
        old = P2C.parse_program
        P2C.parse_program = parse
        try:
            return P2C.facts_only_problems(parent, child, max_words)
        finally:
            P2C.parse_program = old


def world_labels(lineage: str, worlds) -> dict:
    """A fixed random order per lineage: world name -> "world k". Never shown with a rule or family."""
    order = list(worlds)
    random.Random(f"step3-labels-{lineage}").shuffle(order)
    return {w: f"world {k}" for k, w in enumerate(order, 1)}


def format_logs(searches: list, fmt_trial) -> str:
    """searches: [{"label": "world k", "trials": [...], "analyses": {"1": ...}, "base_dev": float}], in label order.
    Each is 2c's per-search log (prog2c.format_logs on that one search) with the heading naming its world, then the
    AI scientist's analysis at each round."""
    parts = []
    for k, s in enumerate(searches, 1):
        one = P2C.format_logs([{"trials": s["trials"]}], fmt_trial, s["base_dev"])
        head = "### Search 1: "
        if not one.startswith(head):
            raise ValueError("prog2c.format_logs changed its heading")
        one = f"### Search {k}, in {s['label']}: " + one[len(head):]
        an = s.get("analyses") or {}
        lines = [f"- round {r}: {an[r]}" for r in sorted(an, key=int)]
        parts.append(one + "\nThe analysis written at each round:\n" + ("\n".join(lines) if lines else "(none)"))
    return "\n\n".join(parts)
