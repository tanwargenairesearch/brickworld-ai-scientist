"""Stage 2c program helpers (PROTOCOL_2C.md): the machine-read header, the facts-only lock, and the rewrite logs,
real or with scores shuffled. Pure functions, no training and no model calls.

- A program is Markdown that starts with a header block; the runner obeys it (Level 3).
- Facts-only rewrites may add a "## Facts" section and nothing else.
- Rewrite logs leave out the per-round analyses in every rewrite (real, facts-only and shuffled alike), because
  they quote real scores in free text. In the shuffled logs, every ok trial's scores move together to another ok trial
  of the same search (a seeded permutation), and score strings quoted in names and hypotheses are relabelled to match.
"""
import json
import random
import re

MAX_RUNS = 30          # rounds x candidates x seeds + retest_top x 2, before confirmation (2b used exactly 30)
RETEST_SEEDS = 2
BOUNDS = {"rounds": (2, 6), "candidates_per_round": (2, 6), "seeds_per_candidate": (1, 3), "retest_top": (1, 4)}
SPLIT_KEYS = ("network", "recipe", "both")
G0_HEADER = {"rounds": 4, "candidates_per_round": 3, "seeds_per_candidate": 2, "retest_top": 3,
             "split": {"network": 0, "recipe": 0, "both": 3}}
_HEADER_RE = re.compile(r"\A\s*<!--\s*header\s*\n(.*?)\n\s*-->\s*\n?", re.S)
_LOOSE_HEADER_RE = re.compile(r"\A\s*<!--(.*?)-->\s*\n?", re.S)


# ---------------------------------------------------------------- header
def render_header(h: dict) -> str:
    sp = h["split"]
    return ("<!-- header\n"
            f"rounds: {h['rounds']}\n"
            f"candidates_per_round: {h['candidates_per_round']}\n"
            f"seeds_per_candidate: {h['seeds_per_candidate']}\n"
            f"retest_top: {h['retest_top']}\n"
            f"split: {{network: {sp['network']}, recipe: {sp['recipe']}, both: {sp['both']}}}\n"
            "-->\n")


def runs_per_search(h: dict) -> int:
    return h["rounds"] * h["candidates_per_round"] * h["seeds_per_candidate"] + h["retest_top"] * RETEST_SEEDS


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
        if k == "split":
            pairs = re.findall(r"(\w+)\s*:\s*(-?\d+)", v)
            h[k] = {a: int(b) for a, b in pairs}
        else:
            if not re.fullmatch(r"-?\d+", v):
                raise ValueError(f"header value for {k} must be a whole number, got {v!r}")
            h[k] = int(v)
    problems = validate_header(h)
    if problems:
        raise ValueError("; ".join(problems))
    return h, text[m.end():]


def strip_header(text: str) -> str:
    """The program without a leading header comment, well-formed or not (for the parent-header fallback)."""
    m = _LOOSE_HEADER_RE.match(text)
    return text[m.end():] if m and "header" in m.group(1) else text.lstrip("\n")


def validate_header(h: dict) -> list:
    out = []
    for k, (lo, hi) in BOUNDS.items():
        if k not in h:
            out.append(f"missing {k}")
        elif not lo <= h[k] <= hi:
            out.append(f"{k} must be {lo}-{hi}, got {h[k]}")
    sp = h.get("split")
    if not isinstance(sp, dict) or set(sp) != set(SPLIT_KEYS):
        out.append("split must give network, recipe and both")
    elif any(v < 0 for v in sp.values()):
        out.append("split counts must be 0 or more")
    elif "candidates_per_round" in h and sum(sp.values()) != h["candidates_per_round"]:
        out.append(f"split must sum to candidates_per_round ({h['candidates_per_round']}), got {sum(sp.values())}")
    extra = set(h) - set(BOUNDS) - {"split"}
    if extra:
        out.append(f"unknown header keys: {sorted(extra)}")
    if not out and runs_per_search(h) > MAX_RUNS:
        out.append(f"rounds x candidates x seeds + retest_top x {RETEST_SEEDS} = {runs_per_search(h)}, over {MAX_RUNS}")
    if not out and h["retest_top"] > h["rounds"] * h["candidates_per_round"]:
        out.append("retest_top can't exceed the number of candidates")
    return out


def g0_program(g0r_text: str) -> str:
    """g0 for Stage 2c: 2b's program body unchanged, with the header 2b effectively ran."""
    return render_header(G0_HEADER) + "\n" + g0r_text


# ---------------------------------------------------------------- facts-only lock
def sections(body: str) -> list:
    """[(heading line or '' for the preamble, full text of the section)], split on '## ' headings."""
    out, cur, buf = [], "", []
    for line in body.splitlines(keepends=True):
        if line.startswith("## "):
            out.append((cur, "".join(buf)))
            cur, buf = line.strip(), [line]
        else:
            buf.append(line)
    out.append((cur, "".join(buf)))
    return out


def facts_only_problems(parent: str, child: str, max_words: int = 1000) -> list:
    """A facts-only rewrite keeps the header and every parent section byte-identical, in order, and may only add one
    '## Facts' section (or append to the parent's). Returns a list of problems (empty if it complies)."""
    try:
        ph, pb = parse_program(parent)
        ch, cb = parse_program(child)
    except ValueError as e:
        return [f"header: {e}"]
    out = []
    if ph != ch:
        out.append("the header changed")
    ps = [s for s in sections(pb) if s[0] != "## Facts"]
    cs = [s for s in sections(cb) if s[0] != "## Facts"]
    norm = lambda t: t.rstrip("\n")  # noqa: E731  trailing blank lines before an added section don't count
    if [(a, norm(b)) for a, b in ps] != [(a, norm(b)) for a, b in cs]:
        pn, cn = [a or "(the text before the first section)" for a, _ in ps], [a or "(the text before the first section)" for a, _ in cs]
        if pn != cn:
            out.append(f"the sections other than '## Facts' must be exactly {pn}, in that order; got {cn}")
        else:
            bad = [pn[i] for i, ((_, b1), (_, b2)) in enumerate(zip(ps, cs)) if norm(b1) != norm(b2)]
            out.append(f"these sections changed and must be copied exactly from the current program: {bad}")
    pf = [b for a, b in sections(pb) if a == "## Facts"]
    cf = [b for a, b in sections(cb) if a == "## Facts"]
    if len(cf) != 1:
        out.append("exactly one '## Facts' section is required")
    elif pf and not cf[0].startswith(pf[0].rstrip("\n")):
        out.append("the parent's '## Facts' text must be kept at the start of the section")
    if len(child.split()) > max_words:
        out.append(f"the program has {len(child.split())} words; the limit is {max_words}")
    return out


# ---------------------------------------------------------------- rewrite logs
def _formats(v: float) -> list:
    return [f"{v:.4f}", f"{v:.3f}", f"{v:.2f}", f"{100 * v:.1f}%", f"{100 * v:.0f}%"]


def _relabel(text: str, mapping: dict) -> str:
    if not mapping or not text:
        return text
    keys = sorted(mapping, key=len, reverse=True)
    pat = re.compile(r"(?<![\d.])(" + "|".join(re.escape(k) for k in keys) + r")(?![\d])")
    return pat.sub(lambda m: mapping[m.group(1)], text)


def shuffle_search(trials: list, seed: int) -> list:
    """A copy of one search's trials with every ok trial's score bundle (dev_exact, dev_per_seed, train_exact)
    permuted among the ok trials. Failures keep their status. Score strings in names and hypotheses are relabelled
    consistently; a short form shared by two different scores is left unrelabelled (disclosed in the report)."""
    trials = json.loads(json.dumps(trials))
    ok = [i for i, t in enumerate(trials) if t.get("result", {}).get("status") == "ok"]
    perm = ok[:]
    random.Random(seed).shuffle(perm)
    fields = ("dev_exact", "dev_per_seed", "train_exact")
    bundles = {i: {f: trials[i]["result"].get(f) for f in fields} for i in ok}
    mapping, clash = {}, set()
    for i, j in zip(ok, perm):
        old, new = bundles[i]["dev_exact"], bundles[j]["dev_exact"]
        if old is not None and new is not None:
            for a, b in zip(_formats(old), _formats(new)):
                if a in mapping and mapping[a] != b:
                    clash.add(a)
                mapping[a] = b
        trials[i]["result"].update(bundles[j])
    mapping = {a: b for a, b in mapping.items() if a not in clash}
    for t in trials:
        t["hypothesis"] = _relabel(t.get("hypothesis", ""), mapping)
        t["name"] = _relabel(t.get("name", ""), mapping)
    return trials


def format_logs(searches: list, fmt_trial, base_dev: float, shuffle_seed=None) -> str:
    """The logs a rewrite reads: per search, the best and selected trial, every trial (no code), and the code of the
    three best. searches: [{"trials": [...]}] as in state.json. fmt_trial: scientist.fmt_trial (passed in, so this
    module stays importable without the frozen code). No per-round analyses, in any arm."""
    parts = []
    for k, st in enumerate(searches):
        trials = st["trials"] if shuffle_seed is None else shuffle_search(st["trials"], 7919 * shuffle_seed + k)
        ok = [t for t in trials if t["result"]["status"] == "ok"]
        best = max(ok, key=lambda t: t["result"]["dev_exact"], default=None)
        sel = best["label"] if best and best["result"]["dev_exact"] > base_dev else None
        lines = [f"### Search {k + 1}: best dev success {best['result']['dev_exact'] if best else None} "
                 f"(starting architecture {base_dev:.3f}); selected: {sel or 'the starting architecture'}", "Trials:"]
        lines += [fmt_trial(t, show_code=False) for t in trials]
        top = sorted(ok, key=lambda t: -t["result"]["dev_exact"])[:3]
        lines.append("Code of the three best trials:")
        lines += [f"{t['label']} ({t['name']}), dev {t['result']['dev_exact']}:\n```python\n{t['code'].strip()}\n```"
                  for t in top]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)
