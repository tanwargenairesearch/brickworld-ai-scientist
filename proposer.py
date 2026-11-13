"""The AI scientist: an LLM that proposes the next training recipes from the history of trials.

Two interchangeable backends send the same system prompt, user message and JSON schema:
  api - Anthropic Messages API (anthropic SDK), structured output via output_config.format
  cli - `claude -p` headless, no tools, no settings, no MCP, in an empty directory
The proposer sees only: the task description, the search space, the per-trial budget, the
baseline's seed spread, and this run's own trials (configs, dev success, training stats).
It never sees the evaluator code, the design sets, other runs, or any final-set result.
"""
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from brickworld.space import BASELINE, SPACE, describe, validate

SYSTEM = (
    "You are an AI research scientist running a bounded, pre-registered experiment campaign. "
    "You choose which training recipes and agent architectures to try next, using the results "
    "so far. Reason carefully about the evidence, including noise between seeds, and propose "
    "the experiments most likely to raise the selection metric within the remaining budget. "
    "Reply only with the JSON object requested."
)

TASK = """## Task
An agent must build a target design in a small brick world.
- The world is a 5 x 5 base, up to 3 layers high. Bricks are one layer high with footprints 1x1, 1x2, 2x1 or 2x2.
- A placed brick drops until it rests on the highest filled cell under its footprint (or the ground), so overhangs with empty cells under them are possible and placement order matters.
- Each step the agent picks a brick type and an anchor cell, or DONE. An episode ends at DONE or after 17 placements. Bricks cannot be removed.
- Target designs are made by 3 to 8 random brick drops. Training uses a stream of fresh random designs; the held-out designs never appear in training.
- The agent observes the current build and the target as described by the obs_mode knob.

## Training and evaluation
- Each trial trains a PPO agent from scratch for exactly 524,288 environment steps on one CPU core. A trial that exceeds 600 seconds of wall-clock time is killed and counts as a failure (larger networks and more PPO epochs cost more time).
- After training, the agent runs greedily on 512 held-out dev designs. A build counts as a success when the project's checker says it matches the target design.
- Selection metric: dev success rate (higher is better). At the end, the trial with the highest dev success is selected. The selected recipe will then be retrained with new seeds and tested once on a separate hidden set of designs, so a lucky seed does not help.
- {noise}

## Search space (every config must set every knob)
{space}

## Baseline config
{baseline}

## Budget
{budget}

## Results so far (this campaign only)
{history}

## What to return
A JSON object with:
- "analysis": at most 120 words on what the results so far suggest.
- "proposals": exactly {k} objects, each with "hypothesis" (one or two sentences: what changes, why it should help, what result would refute it) and "config" (a complete config using only the knobs and values above).
Proposals may repeat an earlier config only to test noise, and should say so."""


def schema() -> dict:
    props = {}
    for k, s in SPACE.items():
        props[k] = {"enum": s["values"]} if s["type"] == "choice" else {"type": "number"}
    cfg = {"type": "object", "properties": props, "required": list(SPACE), "additionalProperties": False}
    item = {"type": "object", "properties": {"hypothesis": {"type": "string"}, "config": cfg},
            "required": ["hypothesis", "config"], "additionalProperties": False}
    return {"type": "object",
            "properties": {"analysis": {"type": "string"}, "proposals": {"type": "array", "items": item}},
            "required": ["analysis", "proposals"], "additionalProperties": False}


def fmt_history(trials: list) -> str:
    if not trials:
        return "No trials yet."
    rows = []
    for t in trials:
        r = t.get("shown", {})
        line = {"trial": t["label"], "round": t["round"], "status": r.get("status"),
                "dev_success": r.get("dev_success"), "train_success_last10pct": r.get("train_success"),
                "wall_s": r.get("wall_s"),
                "params": r.get("params"), "config": t["config"]}
        if t.get("hypothesis"):
            line["your_hypothesis"] = t["hypothesis"]
        rows.append(json.dumps(line))
    return "\n".join(rows)


def build_prompt(trials, round_i, n_rounds, k, noise_note) -> str:
    left = n_rounds - round_i + 1
    budget = (f"This is round {round_i} of {n_rounds}. Each round runs {k} trials in parallel; "
              f"{left * k} trials remain including this round.")
    return TASK.format(noise=noise_note, space=describe(), baseline=json.dumps(BASELINE),
                       budget=budget, history=fmt_history(trials), k=k)


def _parse(obj, k):
    props = obj.get("proposals") if isinstance(obj, dict) else None
    if not isinstance(props, list) or len(props) < k:
        raise ValueError(f"expected {k} proposals, got {0 if not isinstance(props, list) else len(props)}")
    out, errs = [], []
    for i, p in enumerate(props[:k]):
        try:
            out.append({"hypothesis": str(p.get("hypothesis", ""))[:600], "config": validate(p["config"])})
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            errs.append(f"proposal {i + 1}: {e}")
    if errs:
        raise ValueError("; ".join(errs))
    return out, str(obj.get("analysis", ""))[:1500]


class InfraError(RuntimeError):
    """Rate limit, network, server or CLI failure: retried with backoff, never counted as a model attempt."""


class ModelError(RuntimeError):
    """The model's reply was unusable: counts as one of its 3 attempts."""


_INFRA_WORDS = re.compile(r"usage limit|rate.?limit|overloaded|\b429\b|\b5\d\d\b|network|connection|ECONN|"
                          r"timed out|timeout|socket|fetch failed|api error", re.I)


def _loads(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _call_api(model, system, user):
    import anthropic
    client = anthropic.Anthropic(max_retries=6)
    try:
        r = client.messages.create(
            model=model, max_tokens=16000, system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "high", "format": {"type": "json_schema", "schema": schema()}},
        )
    except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError) as e:
        raise InfraError(f"api: {type(e).__name__}: {str(e)[:300]}") from e
    except anthropic.APIStatusError as e:
        if e.status_code in (429, 500, 502, 503, 504, 529):
            raise InfraError(f"api status {e.status_code}: {str(e)[:300]}") from e
        raise
    text = next((b.text for b in r.content if b.type == "text"), "")
    usage = {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens,
             "model": r.model, "stop_reason": r.stop_reason}
    if r.stop_reason == "refusal":
        raise ModelError("the reply was a refusal")
    return _loads(text), text, usage


def _call_cli(model, system, user):
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL")}
    with tempfile.TemporaryDirectory() as d:
        cmd = ["claude", "-p", "--model", model, "--effort", "high", "--tools", "",
               "--strict-mcp-config", "--setting-sources", "", "--disable-slash-commands",
               "--no-session-persistence", "--system-prompt", system,
               "--json-schema", json.dumps(schema()), "--output-format", "json", user]
        try:
            p = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired as e:
            raise InfraError("cli: no reply within 1800 s") from e
    try:
        d = json.loads(p.stdout)
    except json.JSONDecodeError as e:
        raise InfraError(f"cli exit {p.returncode}: {(p.stderr or p.stdout)[-300:]}") from e
    if d.get("is_error"):
        msg = f"cli error ({d.get('subtype')}): {str(d.get('result'))[:300]}"
        raise (InfraError if _INFRA_WORDS.search(msg) else ModelError)(msg)
    obj = d.get("structured_output")
    text = d.get("result") or ""
    if obj is None:
        obj = _loads(text)
    usage = {"input_tokens": (d.get("usage") or {}).get("input_tokens"),
             "cache_creation_input_tokens": (d.get("usage") or {}).get("cache_creation_input_tokens"),
             "output_tokens": (d.get("usage") or {}).get("output_tokens"),
             "models": list((d.get("modelUsage") or {}).keys()),
             "notional_cost_usd": d.get("total_cost_usd")}
    return obj, text or json.dumps(obj), usage


def propose(backend, model, trials, round_i, n_rounds, k, noise_note, log_dir: Path):
    """Return k validated proposals; logs every attempt. Up to 3 attempts with error feedback."""
    log_dir.mkdir(parents=True, exist_ok=True)
    done_file = log_dir / f"round_{round_i:02d}.json"
    if done_file.exists():  # resume: never re-ask a round that already has proposals
        return json.loads(done_file.read_text())["proposals"]
    user = build_prompt(trials, round_i, n_rounds, k, noise_note)
    call = _call_api if backend == "api" else _call_cli
    attempts, infra, feedback, attempt = [], [], "", 0
    fail_file = log_dir / f"round_{round_i:02d}_failed.json"
    while attempt < 3:
        msg = user + (f"\n\nYour previous reply was rejected: {feedback}. Reply again." if feedback else "")
        t0 = time.time()
        raw = None
        try:
            obj, raw, usage = call(model, SYSTEM, msg)
            err = None
        except ModelError as e:
            obj, err = None, str(e)
        except InfraError as e:  # not the model's fault: back off and retry, up to about 6 hours in all
            infra.append({"error": str(e)[:500], "at": time.strftime("%H:%M:%S")})
            fail_file.write_text(json.dumps({"attempts": attempts, "infra_errors": infra}, indent=1))
            if len(infra) >= 40:
                raise RuntimeError(f"proposer infrastructure failed 40 times in round {round_i}: {e}") from e
            time.sleep(min(600, 60 * len(infra)))
            continue
        attempt += 1
        try:
            if err:
                raise ValueError(err)
            props, analysis = _parse(obj, k)
        except (ValueError, TypeError, AttributeError) as e:  # the model's reply was invalid: counts
            feedback = str(e)[:500]
            attempts.append({"attempt": attempt, "error": feedback, "latency_s": round(time.time() - t0, 1),
                             "raw": str(raw)[:8000]})
            fail_file.write_text(json.dumps({"attempts": attempts, "infra_errors": infra}, indent=1))
            time.sleep(20 * attempt)
            continue
        rec = {"backend": backend, "model": model, "round": round_i, "attempt": attempt,
               "latency_s": round(time.time() - t0, 1), "usage": usage, "analysis": analysis,
               "proposals": props, "prompt": msg, "raw": raw, "rejected_attempts": attempts,
               "infra_errors": infra}
        done_file.write_text(json.dumps(rec, indent=1))
        return props
    raise RuntimeError(f"proposer failed 3 times in round {round_i}: {feedback}")


def api_usable() -> bool:
    """True if the API key in the environment has credit (one tiny request)."""
    try:
        import anthropic
        anthropic.Anthropic(max_retries=0).messages.create(
            model="claude-sonnet-5-5", max_tokens=256, output_config={"effort": "low"},
            messages=[{"role": "user", "content": "Reply OK."}])
        return True
    except Exception:  # noqa: BLE001
        return False
