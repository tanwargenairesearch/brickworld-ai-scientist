"""Step 2S prompts (PROTOCOL_STEP2S.md, "The sandbox" and "Arms").

The base prompt is Step 2's, imported unchanged:
  sandbox, scores   Step 2's scores-only prompt (prompts.build("scores", ...)), with Step 2's note
  told_sandbox      Step 2's told task text (the exact rule sentence) with the scores-only feedback (scientist.l1_prompt)
Sandbox arms make one extra call per round, before the proposal call: the base prompt up to its "## What to return"
section, then "## Sandbox" (the protocol's sandbox text, how probes work, earlier probes) and the probe reply format.
The proposal call that follows is the base prompt with "## Sandbox results" inserted before "## What to return".
Probe history: the latest round's probes in full, earlier rounds' probes as their requests only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2s_common as C  # noqa: E402

import prompts as P  # noqa: E402  (Step 2's)
import probes_2s as PB  # noqa: E402

from arch_rsi import scientist as S  # noqa: E402

SANDBOX_TEXT = ("Before you propose this round's candidates, you may run up to 3 probes in this world. A probe places "
                "bricks you choose, one after another, on one of four probe designs, and trains nothing. For each probe "
                "you see what the agent observes at the start and what the world does after each placement. Probes "
                "don't count against your trials. Reply with your probes, or with none.")
HOW = ("How probes work: each probe starts from an empty board on one of the four probe designs, numbered 1 to 4. They "
       "come from the same generator as the training designs, are never scored, and their contents aren't shown. A "
       "placement is a brick (\"1x1\", \"1x2\", \"2x1\" or \"2x2\", rows by columns) with an anchor row and column from "
       f"0 to {PB.SPEC.X - 1}, given as the agent's action gives them. A probe runs up to {C.MAX_PLACEMENTS} placements "
       "in order and ends early if one is illegal, as an episode does. For each probe you get the target's column "
       "heights as the agent observes them at the start, then each placement's filled cells (row y, column x, layer z), "
       "how many of them are inside and outside the target, and its reward, then the final build's column heights. "
       "Height maps list rows y = 0..7 from top to bottom, each a string of columns x = 0..7.")
PROBE_RETURN = (f"A JSON object with \"probes\": a list of at most {C.MAX_PROBES} objects (an empty list runs none), each "
                f"with \"design\" (1 to {C.N_PROBE_DESIGNS}), \"purpose\" (at most 40 words: what the probe should tell "
                f"you) and \"placements\" (1 to {C.MAX_PLACEMENTS} objects, each with \"brick\", \"row\" and \"col\").")
RETURN = "\n## What to return\n"


def base_prompt(arm, world, program, start, trials, incumbent, r, R, K) -> str:
    if arm in ("sandbox", "scores"):
        return P.build("scores", world, program, start, trials, incumbent, r, R, K)
    if arm == "told_sandbox":
        return S.l1_prompt(program, P.task_text("told", world, start), P.interface(), start["code"], start["dev"],
                           trials, incumbent, r, R, K)
    raise ValueError(f"unknown arm {arm}")


def _split(base: str):
    if base.count(RETURN) != 1:
        raise SystemExit("base prompt changed: one '## What to return' section expected")
    head, tail = base.split(RETURN)
    return head, tail


def history_text(hist: dict, latest: int) -> str:
    """hist: {round: {"probes": [...], "results": [...]}}. Round `latest` in full, earlier rounds as requests only."""
    parts = []
    for q in sorted(hist):
        if q == latest:
            parts.append(f"Round {q} probes and what the world did:\n" + PB.render_results(hist[q]["probes"], hist[q]["results"]))
        elif q < latest:
            parts.append(f"Round {q} probes (requests only):\n" + PB.render_requests(hist[q]["probes"]))
    return "\n\n".join(parts)


def probe_prompt(base: str, hist: dict, r: int) -> str:
    head, _ = _split(base)
    earlier = history_text(hist, r - 1) if hist else "No probes yet in this search."
    return (head + "\n## Sandbox\n" + SANDBOX_TEXT + "\n\n" + HOW + "\n\nProbes so far in this search:\n" + earlier
            + RETURN + PROBE_RETURN)


def proposal_prompt(base: str, hist: dict, r: int) -> str:
    head, tail = _split(base)
    return (head + "\n## Sandbox results\nThe probes you asked for this round were run by the harness. They trained "
            "nothing and don't count against your trials.\n\n" + history_text(hist, r) + "\n" + RETURN + tail)


def harness_texts() -> list:
    """Every text the harness writes into a sandbox prompt, for the forbidden-word test."""
    return [SANDBOX_TEXT, HOW, PROBE_RETURN, "Probes so far in this search:", "No probes yet in this search.",
            "The probes you asked for this round were run by the harness. They trained nothing and don't count against "
            "your trials.", "probes and what the world did:", "probes (requests only):", "No probes were run."]
