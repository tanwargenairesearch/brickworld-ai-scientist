#!/bin/zsh
# Waits until Step 2's full run is over, then runs Step 3a (PROTOCOL_STEP3A.md, decision 2: after Step 2, before Step 2S).
# The user approved the launch on 14 November 2026: "Yes, launch then". Run in a Terminal tab under caffeinate -i.
# Over means: Step 2's results.json exists and no Step 2 or Step 2S runner process is alive. run_3a.py checks this again
# itself before anything runs, and refuses if it doesn't hold. Ctrl-C here stops the wait (or the run).
cd /path/to/brickworld-ai-scientist || exit 1
R=arch_rsi/stageH/step2/campaigns/step2/results.json
PAT='python.* arch_rsi/stageH/step2s?/run_step2s?\.py'
echo "[$(date '+%F %T')] waiting for Step 2's full run to finish (checking every 5 minutes)"
while [ ! -f "$R" ] || pgrep -f "$PAT" >/dev/null; do sleep 300; done
sleep 120
echo "[$(date '+%F %T')] Step 2 is over; starting Step 3a"
nice -n 10 .venv/bin/python -I -B -u arch_rsi/stageH/step3a/run_3a.py 2>&1 | tee -a arch_rsi/stageH/step3a/run_stdout.log
echo "[$(date '+%F %T')] Step 3a runner exited with status ${pipestatus[1]}"
