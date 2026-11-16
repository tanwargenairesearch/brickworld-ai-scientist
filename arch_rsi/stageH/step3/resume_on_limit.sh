#!/bin/bash
# Resumes Step 3's full run after stops caused only by the subscription's usage limit.
#
# Why: the frozen call code (proposer.py's _INFRA_WORDS) doesn't recognise "You've hit your 5-hour limit" as an
# infrastructure error, so instead of waiting (PROTOCOL_STEP3.md, "Call limits") the runner spends a search's 3 model
# attempts in under a minute and exits with "N failures; rerun the same command to resume". It first did so on
# 15 Nov 2026 at 21:35 GMT. The user approved this auto-restart the same evening: "Yes, auto-restart".
#
# What it does: reruns the same command (the runner reuses every finished call and training run). After each exit:
#   - results.json with status "final": done;
#   - every failure in that attempt was the usage-limit message: wait (default 20 min) and rerun;
#   - anything else (another failure, a crash, an integrity stop): stop for a human.
# It changes no frozen file and no campaign file. Ctrl-C stops it (and the run).
# Usage: bash arch_rsi/stageH/step3/resume_on_limit.sh            (the real run)
#        STEP3_ROOT=<dir> WAIT_S=1 bash resume_on_limit.sh <cmd...> (tests: a fake root and a fake command)
set -u
ROOT=${STEP3_ROOT:-/path/to/brickworld-ai-scientist}
cd "$ROOT" || exit 1
STEP3=arch_rsi/stageH/step3
CAMP=$STEP3/campaigns/step3
LOG=$STEP3/full_stdout.log
WAIT=${WAIT_S:-1200}
MAX=${MAX_RESTARTS:-30}
if [ $# -gt 0 ]; then CMD=("$@"); else
  CMD=(caffeinate -is nice -n 10 .venv/bin/python -I -B -u $STEP3/run_step3.py --preset step3); fi
ATT=$CAMP/.auto_restart_attempt.log
MARK=$CAMP/.auto_restart_mark
say() { echo "[$(date '+%F %T')] [auto-restart] $*" | tee -a "$LOG"; }
LIMIT='hit your .*limit'

n=0
while :; do
  n=$((n + 1))
  : > "$ATT"; touch "$MARK"; sleep 1
  say "attempt $n: starting the Step 3 runner"
  "${CMD[@]}" 2>&1 | tee -a "$LOG" "$ATT"
  rc=${PIPESTATUS[0]}
  say "attempt $n: runner exited with status $rc"
  if [ -f "$CAMP/results.json" ] && grep -q '"status": "final"' "$CAMP/results.json"; then
    say "results.json is final: Step 3 is done"; exit 0; fi
  if [ -e "$CAMP/INTEGRITY_STOP" ]; then say "STOP: integrity stop (INTEGRITY_STOP exists); needs a human"; exit 2; fi
  if ! tail -n 5 "$ATT" | grep -q 'failures; rerun the same command to resume'; then
    say "STOP: the runner didn't end with resumable failures (crash or other exit); needs a human"; exit 3; fi
  other=$(grep ' failed: ' "$ATT" | grep -v "$LIMIT")
  if [ -n "$other" ]; then
    say "STOP: a failure other than the usage limit; needs a human:"; echo "$other" | head -5 | tee -a "$LOG"; exit 4; fi
  # A call can show the limit last after an earlier genuine bad reply; restarting would give it fresh attempts.
  if ! bad=$(python3 -I - "$CAMP" "$MARK" <<'PY'
import json, os, pathlib, re, sys
camp, mark = pathlib.Path(sys.argv[1]), os.path.getmtime(sys.argv[2])
lim = re.compile(r"hit your .*limit")
bad = []
for f in camp.rglob("*.failed.json"):
    if f.stat().st_mtime < mark or f.with_name(f.name.replace(".failed.json", ".json")).exists():
        continue
    for a in json.loads(f.read_text()).get("attempts", []):
        if not lim.search(str(a.get("error", ""))):
            bad.append(f"{f}: {str(a.get('error', ''))[:150]}")
print("\n".join(bad[:5]))
sys.exit(1 if bad else 0)
PY
  ); then say "STOP: an unfinished call had a bad reply that wasn't the usage limit; needs a human:"; echo "$bad" | tee -a "$LOG"; exit 5; fi
  if [ "$n" -gt "$MAX" ]; then say "STOP: $MAX restarts used; needs a human"; exit 6; fi
  say "every failure in attempt $n was the usage limit ($(grep -c ' failed: ' "$ATT") searches or lineages); waiting $WAIT s, then resuming"
  caffeinate -is sleep "$WAIT"
done
