"""Step 3 prompts (PROTOCOL_STEP3.md).

Search prompts are Step 2's (step2/prompts.py, imported unchanged):
  scores    Step 2's scores-only prompt: 2b's prompt with the note on the world
  readouts  Step 2's readouts prompt (the check arm only, under g0)
With 2 seeds per candidate the prompt is Step 2's build() output exactly. With 1 or 3 seeds, two sentences of the task
text name the seed count, so they are changed to match (task_text3); the rest is unchanged.
The rewrite prompt is adapted from Stage 2c's (L2C_PROMPT in stage2/2c/run_2c.py): Step 3's header, the searches' scores-only
feedback, and new worlds that may each differ from their description. The facts-only restriction is 2c's, word for
word. The no-logs rewrite gets the same prompt with the earlier-generations table and the logs replaced by NONE.
The diagnosis judge's prompt is Step 2's (run_step2.judge_prompt), with rule sentences for the held-out worlds.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3common as C  # noqa: E402
import prog3 as PG  # noqa: E402

import prompts as P  # noqa: E402  (Step 2's, frozen)
from arch_rsi import scientist as S  # noqa: E402

NOTE = P.NOTE
NONE = "None are given for this revision."
RULE_TEXT = {
    "shift": P.RULE_TEXT["shift"],
    "rowshift": ("In this world the agent's observation shows the target shifted: the target, missing, extra and target-height "
                 "channels are rolled 4 rows toward higher y, wrapping around. The world, its designs and scoring are as described."),
    "handshift": ("In this world each placement lands 4 columns to the right of the chosen anchor, wrapping around: a brick "
                  "chosen at column x lands at column (x + 4) mod 8, in the same row. The valid-action mask is in the agent's "
                  "own action terms. The designs and scoring are as described."),
}
RULE_WORDS = P.RULE_WORDS + [r"\bturn\w*", r"\btranspos\w*", r"\bdiagonal\w*", r"\bquarter\w*", r"4 rows"]

_NOISE2 = ("every trial trains 2 seeds and reports the mean dev success. The starting architecture, trained with the "
           "same 2 seeds, scored ")
_SEL2 = ("the trial with the highest 2-seed dev success is selected (the starting architecture if none beats the starting "
         "architecture's own 2-seed score)")


def start_view(start: dict, n: int) -> dict:
    """The start as a search with n seeds per candidate sees it: seeds 0..n-1, rounded as Step 2's run_candidate does."""
    raw = start["raw"][:n]
    return {"code": start["code"], "devs": [round(d, 4) for d in raw], "dev": round(sum(raw) / n, 4),
            "readouts": start.get("readouts")}


def task_text3(arm: str, world: str, start: dict, n: int) -> str:
    t = P.task_text(arm, world, start)
    if n == 2:
        return t
    for part in (_NOISE2, _SEL2):
        if t.count(part) != 1:
            raise SystemExit("Step 2's task text changed: a seed-count sentence was not found")
    if n == 1:
        noise = ("every trial trains 1 seed and reports its dev success. The starting architecture, trained with the "
                 "same seed, scored ")
    else:
        noise = (f"every trial trains {n} seeds and reports the mean dev success. The starting architecture, trained "
                 f"with the same {n} seeds, scored ")
    return t.replace(_NOISE2, noise).replace(_SEL2, _SEL2.replace("2-seed", f"{n}-seed"))


def build3(arm: str, world: str, body: str, start: dict, trials, incumbent, r, R, K, n, readouts=None) -> str:
    """A search prompt. arm: "scores" or "readouts"; body: the program without its header; start: start_view(.., n)."""
    if arm not in ("scores", "readouts"):
        raise ValueError(arm)
    if n == 2:
        return P.build(arm, world, body, start, trials, incumbent, r, R, K, readouts=readouts)
    task = task_text3(arm, world, start, n)
    if arm == "scores":
        return S.l1_prompt(body, task, P.interface(), start["code"], start["dev"], trials, incumbent, r, R, K)
    return P.l1_prompt_readouts(body, task, P.interface(), start["code"], start["dev"], trials, incumbent, r, R, K,
                                start_readouts=start.get("readouts"), readouts=readouts)


L3_PROMPT = """## Your current research program ({name})
{program}

## How the program is used
The program has two parts.
- The header at the top (the HTML comment that starts with "<!-- header") is read only by the code that runs each search. It sets the number of rounds, the candidates per round and the training seeds per candidate. Bounds: rounds 2-6; candidates_per_round 2-6; seeds_per_candidate 1-3; and rounds x candidates_per_round x seeds_per_candidate must be at most {max_runs}, so every program gets the same number of training runs per search. After the search, its selected trial is retrained on 4 new seeds, and that is its score. Keep the header's format and change only its numbers.
- The rest of the program is what you receive at the start of every round of every architecture search, together with a fixed task description (which states the header's settings in words), the interface rules, the starting architecture, the search's results so far (every trial's code, hypothesis and scores) and the budget. The searches see scores only: dev success per seed, training success, parameters, training compute, steps and time.

The program is the only part of the procedure you can change. Next it will be used for new searches in brick worlds that may each differ from their description in ways you have not seen. It will be judged by how well the architectures found under it do there, retrained on new seeds, at the same training budget.

The interface rules, caps, checks, the training budget and the recipe bounds are fixed and are not part of the program. In particular the policy and its features must be learned: a program that advises hand-writing a placement rule or score, hand-computing features from the observation before a learned layer, hand-setting weights so the network is competent before training, or ways to pass the checks without learning, is advising rule violations. Candidates built that way are rejected by the checks or reported as violations after the campaign.
{restriction}
## The fixed task description, as given in the searches
{task_example}

## Earlier generations
{gen_table}

## Logs of the searches run under {logs_label}
{logs}

## What to return
{instruction} Return a JSON object with:
- "lessons": at most 200 words: what these searches show about which architectures, recipes and ways of searching worked.
- "changes": the changes you are making to the program, each with the evidence for it.
- "program": the complete new program in Markdown, starting with the header, at most {max_words:,} words.
- "expected_effect": one sentence: what should improve, and how one could tell."""

def logs_intro(n: int = 7) -> str:
    where = "labelled world 1" if n == 1 else f"labelled world 1 to world {n} (the same label means the same world)"
    return f"Each search ran in one brick world, {where}. Each world may differ from its description in its own way."


LOGS_INTRO = logs_intro(len(C.TRAIN_WORLDS))


def facts_restriction() -> str:
    """2c's FACTS_RESTRICTION, word for word (test_prompts3.py compares it with run_2c.py's)."""
    return ("\n## A restriction on this revision\nIn this revision you may only add facts. Return the current program exactly "
            "as it is, character for character, including its header and every section, and add one section at the end headed "
            "\"## Facts\": a short list of what these searches showed (for example which kinds of network or recipe did well or "
            "badly, and how large the differences were compared with seed noise). Do not change, move or remove anything else. "
            "A program that changes anything other than adding that section is rejected.\n")


def l3_prompt(kind: str, name: str, program: str, task_example: str, gen_table: str, logs_label: str, logs: str,
              max_words: int = C.MAX_WORDS, n_worlds: int = len(C.TRAIN_WORLDS)) -> str:
    """kind: "real" (g1, g2, g3), "facts" or "nologs"."""
    if kind not in ("real", "facts", "nologs"):
        raise ValueError(kind)
    nolog = kind == "nologs"
    return L3_PROMPT.format(
        name=name, program=program.rstrip("\n"), max_runs=PG.MAX_TRIAL_RUNS, restriction=facts_restriction() if kind == "facts" else "",
        task_example=task_example, gen_table=NONE if nolog else gen_table, logs_label=logs_label,
        logs=NONE if nolog else logs_intro(n_worlds) + "\n\n" + logs, max_words=max_words,
        instruction=("Follow the restriction above." if kind == "facts"
                     else "Revise the program by following its own section on revising it."))


def judge_prompt(world: str, analyses: list, name: str, hypothesis: str) -> str:
    """Step 2's judge prompt (run_step2.judge_prompt) with this world's rule sentence."""
    rounds = "\n\n".join(f"### Round {r}\n{a}" for r, a in enumerate(analyses, 1))
    return f"""A simulated brick world had this hidden rule:
{RULE_TEXT[world]}

Below is what an AI scientist wrote during a search in that world: its analysis at every round, in order, and the
name and hypothesis of the trial it ended with. Does the text, anywhere, state the rule?
- "yes": it states the rule's substance (what is changed and how).
- "partly": it identifies the right kind of change but not its specifics.
- "no": anything else.

## Analysis at each round
{rounds}

## Trial it ended with
name: {name}
hypothesis: {hypothesis}

Return a JSON object: "verdict" ("yes", "partly" or "no") and "evidence" (at most 80 words, quoting the text and
naming the round it comes from)."""


def rule_words(text: str) -> list:
    return [m.group(0) for w in RULE_WORDS for m in re.finditer(w, text, re.I)]
