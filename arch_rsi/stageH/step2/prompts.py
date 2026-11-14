"""Step 2 prompts (PROTOCOL_STEP2.md, "Arms" and "Readouts").

Every arm gets 2b's task text (TASK_2B and RECIPE_TEXT["both"] from stage2/2b/run_2b.py, filled in for W_s, with the
selection sentence changed to Step 2's selection), the interface text, program g0r.md and the starting architecture,
plus one section that differs by arm:
  told      "## This world's hidden rule" with the rule sentence below; scores and readouts
  readouts  "## A note on this world" with NOTE; scores and readouts
  scores    the same note; 2b's prompt (scientist.l1_prompt), unchanged
  hidden    the same note; Stage 1's hidden-score prompt (run_stage1.l1_prompt_hidden), unchanged
l1_prompt_readouts is scientist.l1_prompt with readouts added after the starting code and after each trial; with no
readouts it returns exactly scientist.l1_prompt's text (test_prompts_step2.py).
"""
import importlib.util
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as C  # noqa: E402

from arch_rsi import scientist as S  # noqa: E402
from arch_rsi.interface import INTERFACE_TEXT  # noqa: E402
from arch_rsi.worlds import CAPS  # noqa: E402
import readouts as RO  # noqa: E402

_spec = importlib.util.spec_from_file_location("run_2b", C.ROOT / "arch_rsi" / "stage2" / "2b" / "run_2b.py")
R2B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R2B)
S1 = R2B.S1

NOTE = "This world may not behave exactly as described above. Nothing more about it is stated."
RULE_TEXT = {
    "shift": ("In this world the agent's observation shows the target shifted: the target, missing, extra and target-height "
              "channels are rolled 4 columns toward higher x, wrapping around. The world, its designs and scoring are as described."),
    "mirror": ("In this world each placement lands at the left-right mirror of the chosen anchor: a brick dx cells wide chosen at "
               "column x lands at column 8 - dx - x, in the same row. The valid-action mask is in the agent's own action terms. "
               "The designs and scoring are as described."),
}
OLD_SEL = ("At the end of the search, the trial with the highest dev success is selected. It is then retrained with new seeds "
           "and tested once on a separate hidden set of designs, so a lucky seed does not help.")
NEW_SEL = ("At the end of the search, the trial with the highest 2-seed dev success is selected (the starting architecture if "
           "none beats the starting architecture's own 2-seed score). It is then retrained with 4 new seeds and scored on "
           "the dev designs again, so a lucky seed does not help.")
RULE_WORDS = [r"\bshift\w*", r"\bmirror\w*", r"\bflip\w*", r"\broll(ed|s|ing)?\b", r"\boffset\w*", r"\breflect\w*",
              r"\bwrap\w*", r"\brotat\w*", r"4 columns"]


def interface() -> str:
    return INTERFACE_TEXT.format(max_params=CAPS["max_params"], flops_ratio=CAPS["flops_ratio"], n_check=CAPS["n_check"],
                                 short_steps=CAPS["short_steps"], max_untrained=CAPS["max_untrained_exact"],
                                 max_clean=CAPS["max_untrained_clean"], max_short=CAPS["max_short_exact"])


def task_text(arm: str, world: str, start: dict, steps: int = 131072, timeout: int = 900) -> str:
    sp = C.common.register()
    if arm == "hidden":
        noise = "every trial trains 2 seeds. Training runs on this task can succeed or fail by seed luck alone."
    else:
        per = ", ".join(f"{d:.3f}" for d in start["devs"])
        noise = (f"every trial trains 2 seeds and reports the mean dev success. The starting architecture, trained with the "
                 f"same 2 seeds, scored {per} on dev (mean {start['dev']:.3f}): training runs on this task can succeed or "
                 f"fail by seed luck alone, so treat single differences of this size with care.")
    t = R2B.TASK_2B.format(X=sp.X, Y=sp.Y, Z=sp.Z, max_steps=sp.max_steps, k_min=sp.k_min, k_max=sp.k_max, steps=steps,
                           timeout=timeout, n_dev=512, noise=noise,
                           recipe_text=R2B.RECIPE_TEXT["both"].replace("{{", "{").replace("}}", "}"))
    if t.count(OLD_SEL) != 1:
        raise SystemExit("task text changed: selection sentence not found")
    t = t.replace(OLD_SEL, NEW_SEL)
    if arm == "told":
        return t + "\n## This world's hidden rule\n" + RULE_TEXT[world] + "\n"
    return t + "\n## A note on this world\n" + NOTE + "\n"


def l1_prompt_readouts(program, task, interface_text, base_code, base_dev, trials, incumbent, r, R, K,
                       start_readouts=None, readouts=None) -> str:
    """scientist.l1_prompt with readouts. readouts: {label: readout dict}; full text for the incumbent and round r-1."""
    readouts = readouts or {}

    def one(t):
        s = S.fmt_trial(t)
        ro = readouts.get(t["label"])
        if ro:
            full = t["round"] == r - 1 or (incumbent is not None and t["label"] == incumbent["label"])
            s += "\nreadouts (seed 0):\n" + (ro["text_full"] if full else ro["text_short"])
        return s
    hist = "No trials yet." if not trials else "\n\n".join(one(t) for t in trials)
    inc = "the starting architecture" if incumbent is None else f"trial {incumbent['label']} ({incumbent.get('name')})"
    left = (R - r + 1) * K
    start_ro = ("" if not start_readouts else
                "\n\n## Readouts\n" + RO.LEGEND + "\n\nReadouts of the starting architecture (seed 0):\n" + start_readouts["text_full"])
    return f"""## Your research program
{program}

{task}
{interface_text}

## Starting architecture (dev success {base_dev:.3f})
```python
{base_code.strip()}
```{start_ro}

## Results so far in this search
Current best: {inc}.

{hist}

## Budget
This is round {r} of {R}. Each round trains {K} candidates in parallel; {left} trials remain including this round.

## What to return
A JSON object with:
- "analysis": at most 150 words on what the results so far suggest.
- "proposals": exactly {K} objects, each with "name" (a few words), "hypothesis" (at most 80 words: what changes, why it should help, what result would refute it) and "code" (the complete candidate file)."""


def build(arm, world, program, start, trials, incumbent, r, R, K, readouts=None) -> str:
    task = task_text(arm, world, start)
    itf = interface()
    if arm == "hidden":
        return S1.l1_prompt_hidden(program, task, itf, start["code"], trials, r, R, K)
    if arm == "scores":
        return S.l1_prompt(program, task, itf, start["code"], start["dev"], trials, incumbent, r, R, K)
    return l1_prompt_readouts(program, task, itf, start["code"], start["dev"], trials, incumbent, r, R, K,
                              start_readouts=start.get("readouts"), readouts=readouts)


def rule_words(text: str) -> list:
    return [m.group(0) for w in RULE_WORDS for m in re.finditer(w, text, re.I)]
