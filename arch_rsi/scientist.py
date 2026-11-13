"""The AI scientist at two levels, both Claude Opus 5.5 via `claude -p` (no tools, no settings, no MCP,
run in an empty directory).

Level 1 (architecture search): given its research program, the task, the interface, the starting
architecture and this search's results (every trial's code, hypothesis and scores), it writes K new
candidate architecture files. Files that fail the pre-training checks are returned with the error for
up to 2 repair calls.
Level 2 (self-revision): given its current research program, which contains the rules for revising
itself, and the logs of the searches run under it, it rewrites the whole program.

The scientist never sees the evaluator, the design sets, other searches' results (Level 1), or any
final-set score.
"""
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from proposer import _INFRA_WORDS, InfraError, ModelError  # frozen main-campaign module, reused unchanged

MODEL = "claude-opus-5-5"
EFFORT = "high"

SYSTEM_L1 = ("You are an AI research scientist searching for neural-network architectures for a reinforcement-"
             "learning agent. You write complete, correct PyTorch files. Follow your research program, reason "
             "carefully about the evidence including noise between seeds, and propose the experiments most likely "
             "to find a better architecture within the remaining budget. Reply only with the JSON object requested.")

SYSTEM_L2 = ("You are an AI research scientist improving your own research program: the instructions you follow "
             "whenever you search for neural-network architectures. Reason carefully about what the evidence "
             "supports. Reply only with the JSON object requested.")

TASK = """## Task
An agent must build a target design in a brick world.
- The world is a {X} x {Y} base, up to {Z} layers high. Bricks are one layer high with footprints 1x1, 1x2, 2x1 or 2x2.
- A placed brick drops until it rests on the highest filled cell under its footprint (or the ground), so overhangs with empty cells under them are possible and placement order matters.
- Each step the agent picks a brick type and an anchor cell, or DONE. An episode ends at DONE or after {max_steps} placements. Bricks cannot be removed.
- Target designs are made by {k_min} to {k_max} random brick drops. Training uses a stream of fresh random designs; the held-out designs never appear in training.

## Training and evaluation (fixed: you change only the network)
- Compute-matched budget: each trial trains a PPO agent from scratch on one CPU core with the same training compute as the starting architecture gets in {steps:,} environment steps. A network whose training FLOPs per sample are r times the starting architecture's trains for {steps:,} / r steps (rounded down to whole updates of 1,024 steps); a cheaper network still gets {steps:,} steps. So a bigger network sees fewer designs. The recipe is fixed: Adam with learning rate 0.001 decaying linearly to 0, 64 parallel worlds, 16-step rollouts, 4 PPO epochs of 8 minibatches, gamma 0.95, GAE lambda 0.9, clip 0.2, entropy coefficient 0.01, value coefficient 0.5, gradient-norm clip 1.0, normalised advantages. Reward per placement: (target cells filled minus cells filled outside the target) / the target's cell count, minus 0.01; plus 1 when the episode ends with an exact build.
- A trial that runs longer than {timeout:,} seconds is killed and counts as a failure. Bigger or slower networks take longer.
- After training, the agent runs greedily on {n_dev} held-out dev designs. Success means the finished build equals the target design exactly, cell for cell.
- Selection metric: dev success (higher is better). At the end of the search, the trial with the highest dev success is selected. It is then retrained with new seeds and tested once on a separate hidden set of designs, so a lucky seed does not help.
- Noise: {noise}
"""


def task_text(world, steps, timeout, n_dev, noise) -> str:
    return TASK.format(X=world.X, Y=world.Y, Z=world.Z, max_steps=world.max_steps, k_min=world.k_min,
                       k_max=world.k_max, steps=steps, timeout=timeout, n_dev=n_dev, noise=noise)


def fmt_trial(t: dict, show_code: bool = True) -> str:
    r = t.get("result", {})
    head = {"trial": t["label"], "round": t["round"], "name": t.get("name"), "status": r.get("status"),
            "dev_success": r.get("dev_exact"), "dev_success_per_seed": r.get("dev_per_seed"),
            "train_success_last10pct": r.get("train_exact"),
            "params": r.get("params"), "train_flops_vs_start": r.get("flops_ratio"), "train_steps": r.get("train_steps"),
            "train_seconds": r.get("wall_s")}
    if r.get("error"):
        head["error"] = str(r["error"])[:400]
    s = json.dumps(head) + "\nhypothesis: " + str(t.get("hypothesis", ""))
    if show_code and t.get("code"):
        s += "\n```python\n" + t["code"].strip() + "\n```"
    return s


def l1_prompt(program, task, interface, base_code, base_dev, trials, incumbent, r, R, K) -> str:
    hist = "No trials yet." if not trials else "\n\n".join(fmt_trial(t) for t in trials)
    inc = "the starting architecture" if incumbent is None else f"trial {incumbent['label']} ({incumbent.get('name')})"
    left = (R - r + 1) * K
    return f"""## Your research program
{program}

{task}
{interface}

## Starting architecture (dev success {base_dev:.3f})
```python
{base_code.strip()}
```

## Results so far in this search
Current best: {inc}.

{hist}

## Budget
This is round {r} of {R}. Each round trains {K} candidates in parallel; {left} trials remain including this round.

## What to return
A JSON object with:
- "analysis": at most 150 words on what the results so far suggest.
- "proposals": exactly {K} objects, each with "name" (a few words), "hypothesis" (at most 80 words: what changes, why it should help, what result would refute it) and "code" (the complete candidate file)."""


def l1_schema(k: int) -> dict:
    item = {"type": "object", "properties": {"name": {"type": "string"}, "hypothesis": {"type": "string"},
                                             "code": {"type": "string"}},
            "required": ["name", "hypothesis", "code"], "additionalProperties": False}
    return {"type": "object", "properties": {"analysis": {"type": "string"},
                                             "proposals": {"type": "array", "items": item}},
            "required": ["analysis", "proposals"], "additionalProperties": False}


L2_SCHEMA = {"type": "object", "properties": {
    "lessons": {"type": "string"},
    "changes": {"type": "array", "items": {"type": "object", "properties": {
        "change": {"type": "string"}, "evidence": {"type": "string"}},
        "required": ["change", "evidence"], "additionalProperties": False}},
    "program": {"type": "string"}, "expected_effect": {"type": "string"}},
    "required": ["lessons", "changes", "program", "expected_effect"], "additionalProperties": False}


def call_cli(system: str, user: str, schema: dict, timeout: int = 2700):
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL")}
    with tempfile.TemporaryDirectory() as d:
        cmd = ["claude", "-p", "--model", MODEL, "--effort", EFFORT, "--tools", "", "--strict-mcp-config",
               "--setting-sources", "", "--disable-slash-commands", "--no-session-persistence",
               "--system-prompt", system, "--json-schema", json.dumps(schema), "--output-format", "json"]
        try:
            p = subprocess.run(cmd, cwd=d, env=env, input=user, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise InfraError(f"cli: no reply within {timeout} s") from e
    try:
        out = json.loads(p.stdout)
    except json.JSONDecodeError as e:
        raise InfraError(f"cli exit {p.returncode}: {(p.stderr or p.stdout)[-300:]}") from e
    if out.get("is_error"):
        msg = f"cli error ({out.get('subtype')}): {str(out.get('result'))[:300]}"
        raise (InfraError if _INFRA_WORDS.search(msg) else ModelError)(msg)
    obj = out.get("structured_output")
    text = out.get("result") or ""
    if obj is None:
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            obj = None
    usage = {"input_tokens": (out.get("usage") or {}).get("input_tokens"),
             "cache_read_input_tokens": (out.get("usage") or {}).get("cache_read_input_tokens"),
             "output_tokens": (out.get("usage") or {}).get("output_tokens"),
             "models": list((out.get("modelUsage") or {}).keys()), "notional_cost_usd": out.get("total_cost_usd")}
    return obj, text or json.dumps(obj), usage


def call_with_retries(system, user, schema, validate, log_path: Path, max_model_attempts: int = 3):
    """Call until validate(obj) passes. Infra errors back off (up to ~6 h); invalid replies count as attempts."""
    attempts, infra, feedback, n = [], [], "", 0
    while n < max_model_attempts:
        msg = user + (f"\n\nYour previous reply was rejected: {feedback}. Reply again in full." if feedback else "")
        t0 = time.time()
        try:
            obj, raw, usage = call_cli(system, msg, schema)
            err = None
        except ModelError as e:
            obj, raw, usage, err = None, "", {}, str(e)
        except InfraError as e:
            infra.append({"error": str(e)[:500], "at": time.strftime("%Y-%m-%d %H:%M:%S")})
            log_path.with_suffix(".failed.json").write_text(json.dumps({"attempts": attempts, "infra": infra}, indent=1))
            if len(infra) >= 40:
                raise RuntimeError(f"infrastructure failed 40 times: {e}") from e
            time.sleep(min(600, 60 * len(infra)))
            continue
        n += 1
        try:
            if err:
                raise ValueError(err)
            val = validate(obj)
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            feedback = str(e)[:600]
            attempts.append({"attempt": n, "error": feedback, "latency_s": round(time.time() - t0, 1), "raw": str(raw)[:20000]})
            log_path.with_suffix(".failed.json").write_text(json.dumps({"attempts": attempts, "infra": infra}, indent=1))
            time.sleep(15 * n)
            continue
        rec = {"model": MODEL, "effort": EFFORT, "attempt": n, "latency_s": round(time.time() - t0, 1), "usage": usage,
               "prompt": msg, "reply": obj, "rejected_attempts": attempts, "infra_errors": infra,
               "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        log_path.write_text(json.dumps(rec, indent=1))
        return val, rec
    raise RuntimeError(f"model failed {max_model_attempts} times: {feedback}")


def parse_proposals(obj, k):
    props = obj.get("proposals") if isinstance(obj, dict) else None
    if not isinstance(props, list) or len(props) != k:
        raise ValueError(f"expected exactly {k} proposals, got {0 if not isinstance(props, list) else len(props)}")
    out = []
    for i, p in enumerate(props):
        code = p.get("code")
        if not isinstance(code, str) or "def build_policy" not in code:
            raise ValueError(f"proposal {i + 1} has no build_policy function in its code")
        out.append({"name": str(p.get("name", ""))[:80], "hypothesis": str(p.get("hypothesis", ""))[:1000],
                    "code": code})
    return {"analysis": str(obj.get("analysis", ""))[:2000], "proposals": out}


def repair_prompt(base_prompt: str, failed: list) -> str:
    parts = [base_prompt, "\n\n## Some of your proposals failed the pre-training checks"]
    for f in failed:
        parts.append(f"### {f['name']}\nerror: {f['error']}\n```python\n{f['code'].strip()}\n```")
    parts.append(f"Return corrected versions of exactly these {len(failed)} proposals, in the same order and the same "
                 f"JSON format (\"analysis\" may be one sentence). Keep each one's idea unless the error shows it "
                 f"cannot work within the rules; then say so in its hypothesis.")
    return "\n\n".join(parts)


def l2_prompt(g, program, R, K, task_example, gen_table, logs, max_words) -> str:
    return f"""## Your current research program (generation {g})
{program}

## How the program is used
At the start of every round of every architecture search you receive this program together with a fixed task description, the interface rules, the starting architecture, the search's results so far (every trial's code, hypothesis and scores) and the budget. Each search has {R} rounds of {K} candidates and starts from the same starting architecture. The program is the only part of the procedure you can change. Next it will be used for new searches, including on brick worlds of other sizes that you will not see while revising it. It will be judged by how well the architectures found under it do on hidden designs, at the same budget.

The interface rules, caps, checks, budget and recipe are fixed and are not part of the program. In particular the policy and its features must be learned: a program that advises hand-writing a placement rule or score, hand-computing features from the observation before a learned layer, hand-setting weights so the network is competent before training, or ways to pass the checks without learning, is advising rule violations. Candidates built that way are rejected by the checks or reported as violations after the campaign.

## The fixed task description, as given in this generation's searches
{task_example}

## Earlier generations
{gen_table}

## Logs of the searches run under generation {g}
{logs}

## What to return
Revise the program by following its own section on revising it. Return a JSON object with:
- "lessons": at most 200 words: what these searches show about which architectures and which ways of searching worked.
- "changes": the changes you are making to the program, each with the evidence for it.
- "program": the complete new program in Markdown, at most {max_words:,} words.
- "expected_effect": one sentence: what should improve, and how one could tell."""


def parse_program(obj, max_words):
    prog = obj.get("program") if isinstance(obj, dict) else None
    if not isinstance(prog, str) or len(prog.split()) < 50:
        raise ValueError("program missing or shorter than 50 words")
    if len(prog.split()) > max_words:
        raise ValueError(f"program has {len(prog.split())} words; the limit is {max_words}")
    return {"program": prog.strip() + "\n", "lessons": str(obj.get("lessons", ""))[:3000],
            "changes": obj.get("changes", []), "expected_effect": str(obj.get("expected_effect", ""))[:600]}
