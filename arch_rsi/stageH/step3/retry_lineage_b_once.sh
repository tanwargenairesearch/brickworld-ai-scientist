#!/bin/bash
# Retries lineage B's g3 rewrite once, after Step 3's auto-restart stops for it.
#
# Why: at 00:46 GMT on 16 Nov 2026 all 3 attempts at lineage B's g3 were over the 1,000-word limit (1,250, 1,094 and
# 1,019 words). The runner logged "lineage B failed" at 05:10 and carried on, so it will end with "1 failures; rerun the
# same command to resume", and resume_on_limit.sh stops there for a human (RUN_NOTES.md, section 3). The user approved
# retrying it automatically on 16 Nov 2026: "Yes, retry automatically".
#
# What it does:
#   1. waits for the running resume_on_limit.sh (WATCH_PID) to exit;
#   2. retries only if the stop was for exactly this: results.json isn't final, there's no INTEGRITY_STOP, the runner
#      ended with resumable failures, the last attempt's only failure that isn't the usage limit is lineage B's
#      word-limit one, and the only unfinished call with a bad reply is programs/B/call_g3, every bad reply being the
#      word limit;
#   3. if so, runs resume_on_limit.sh once more (it still waits out usage-limit stops); otherwise it stops and says why.
# It retries once at most: if B's g3 fails again, the auto-restart stops for a human as before.
# Usage: WATCH_PID=<pid of resume_on_limit.sh> bash arch_rsi/stageH/step3/retry_lineage_b_once.sh
#        tests: STEP3_ROOT=<dir> WATCH_PID=<pid> RUNNER_PAT=<none> POLL_S=1 bash retry_lineage_b_once.sh <cmd...>
set -u
ROOT=${STEP3_ROOT:-/path/to/brickworld-ai-scientist}
cd "$ROOT" || exit 1
STEP3=arch_rsi/stageH/step3
CAMP=$STEP3/campaigns/step3
LOG=$STEP3/full_stdout.log
ATT=$CAMP/.auto_restart_attempt.log
MARK=$CAMP/.auto_restart_mark
POLL=${POLL_S:-60}
RUNNER_PAT=${RUNNER_PAT:-run_step3.py --preset step3}
LIMIT='hit your .*limit'
BLINE='lineage B failed: RuntimeError: model failed 3 times: program has [0-9]+ words; the limit is 1000$'
if [ $# -gt 0 ]; then RERUN=("$@"); else RERUN=(/bin/bash "$STEP3/resume_on_limit.sh"); fi
say() { echo "[$(date '+%F %T')] [retry-B-g3] $*" | tee -a "$LOG"; }
stop() { say "NOT RETRYING: $*; needs a human"; exit 1; }

PID=${WATCH_PID:?set WATCH_PID to the pid of the running resume_on_limit.sh}
kill -0 "$PID" 2>/dev/null || { echo "process $PID isn't running"; exit 1; }
say "watching the auto-restart (pid $PID); will retry lineage B's g3 once if the run stops only for it"
while kill -0 "$PID" 2>/dev/null; do sleep "$POLL"; done
sleep 2
say "the auto-restart (pid $PID) has exited; checking why"

if [ -f "$CAMP/results.json" ] && grep -q '"status": "final"' "$CAMP/results.json"; then
  say "results.json is final: Step 3 is done; nothing to retry"; exit 0; fi
[ -e "$CAMP/INTEGRITY_STOP" ] && stop "integrity stop (INTEGRITY_STOP exists)"
tail -n 5 "$ATT" | grep -q 'failures; rerun the same command to resume' || stop "the runner didn't end with resumable failures"
other=$(grep ' failed: ' "$ATT" | grep -v "$LIMIT")
[ "$(printf '%s\n' "$other" | grep -c .)" -eq 1 ] || { printf '%s\n' "$other" | head -5 | tee -a "$LOG"; stop "the non-limit failures aren't exactly one"; }
printf '%s\n' "$other" | grep -Eq "$BLINE" || { printf '%s\n' "$other" | tee -a "$LOG"; stop "the one non-limit failure isn't lineage B's word limit"; }
if ! bad=$(python3 -I - "$CAMP" "$MARK" <<'PY'
import json, os, pathlib, re, sys
camp, mark = pathlib.Path(sys.argv[1]), os.path.getmtime(sys.argv[2])
lim = re.compile(r"hit your .*limit")
words = re.compile(r"^program has \d+ words; the limit is 1000$")
target = camp / "programs" / "B" / "call_g3.failed.json"
seen, bad = False, []
for f in camp.rglob("*.failed.json"):
    if f.stat().st_mtime < mark or f.with_name(f.name.replace(".failed.json", ".json")).exists():
        continue
    for a in json.loads(f.read_text()).get("attempts", []):
        e = str(a.get("error", ""))
        if lim.search(e):
            continue
        if f == target and words.search(e):
            seen = True
        else:
            bad.append(f"{f}: {e[:150]}")
if not seen:
    bad.append(f"{target}: no word-limit failure found")
print("\n".join(bad[:5]))
sys.exit(1 if bad else 0)
PY
); then printf '%s\n' "$bad" | tee -a "$LOG"; stop "an unfinished call has a bad reply other than B's g3 word limit"; fi
pgrep -f "$RUNNER_PAT" >/dev/null && stop "a Step 3 runner is still running"

say "the stop was only lineage B's g3 word limit: retrying once (user, 16 Nov 2026: \"Yes, retry automatically\")"
"${RERUN[@]}"
rc=$?
say "the retried auto-restart exited with status $rc; no further retries"
exit "$rc"
