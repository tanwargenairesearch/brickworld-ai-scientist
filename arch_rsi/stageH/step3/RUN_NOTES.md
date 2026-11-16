# Step 3 full run: notes for the report

Events during the full run that `REPORT_STEP3.md` must state. Times are GMT.

## 15 Nov 2026: usage-limit stop, then auto-restart

- **What happened:** from about 21:11, every call to the AI scientist returned "You've hit your 5-hour limit · resets 9:40pm (Europe/London)".
- **Why the run stopped instead of waiting:**
  - The frozen call code (`proposer.py`, `_INFRA_WORDS`) matches "usage limit" and "rate limit", not "5-hour limit". So each reply counted as a bad model reply, not an infrastructure error.
  - Each affected search used its 3 model attempts in under a minute: 96 failures with this message, from 21:11 to 21:35.
  - The runner then exited at 21:35 with "97 failures; rerun the same command to resume". The 97th failure is below.
  - `PROTOCOL_STEP3.md` ("Call limits") expected the runner to wait and retry.
- **Data:**
  - No saved reply or saved prompt contains the limit message. Checked over every campaign JSON file except results, states and training logs.
  - Every finished call and every training run is reused on resume. 1,239 training runs were done at the stop, all ok.
- **Restart:**
  - Approved by the user at about 22:05: "Yes, auto-restart".
  - `resume_on_limit.sh` reruns the same command. It restarts only when every failure in an attempt was the limit message, and stops for a human on anything else. It changes no frozen or campaign file.
  - Attempt 1 started at 22:08:58. Preflight passed: all 128 hashes unchanged.
  - Every later restart is logged in `full_stdout.log` with the prefix `[auto-restart]`.

## Lineage A's second rewrite (g2): more than 3 attempts

- **First try, at 20:50:58:** all 3 attempts exceeded the 1,000-word limit (1,167, 1,051 and 1,143 words; `programs/A/call_g2.failed.json`). The runner stopped lineage A; the rest of the run went on.
- **On resume, at 22:12:30:**
  - The call got fresh attempts, because the runner repeats only unfinished calls.
  - Attempt 1 had 1,150 words, and attempt 2 was valid at 942 words (`programs/A/call_g2.json`).
  - So g2 took 5 attempts in all, against the protocol's 3.
- **What this can affect:** only lineage A's g2 and later rewrites, so P4 (final − g1) and the descriptive tables. P1, P2 and P3 use g1, facts-only and no-logs, which were all written before this.

## 3. Lineage B's g3 rewrite failed the word limit (16 Nov, 05:10 GMT)

- **What happened:** all 3 attempts at B's g3 were over 1,000 words (1,250, 1,094, 1,019; `programs/B/call_g3.failed.json`). The runner logged "lineage B failed" at 05:10:16 and carried on with every other search.
- **Effect on this pass:** B has no g3 and no final-arm searches yet. The runner will finish everything else, then exit with "1 failures; rerun the same command to resume". The auto-restart script stops there for the user, because this failure isn't the usage limit.
- **The protocol:** it sets the 1,000-word limit and says final is the last kept program, but not what happens when a rewrite can't be written. The runner's design is to rerun, which makes a fresh call (3 more attempts), as with A's g2.
- **Decision:** the user approved retrying it automatically on 16 Nov 2026 ("Yes, retry automatically"). Rerunning gives B's g3 more than the protocol's 3 attempts. Either way it can affect only P4 and the descriptive tables, because P1–P3 use g1, facts-only and no-logs.
  - **How:** `retry_lineage_b_once.sh`, started at 09:22:18 GMT in the Terminal tab "Step 3 retry B g3 once", waits for the auto-restart (pid 98810) to exit. It reruns `resume_on_limit.sh` once, and only if the sole non-limit failure is B's g3 word limit. Its log lines are tagged `[retry-B-g3]` in `full_stdout.log`. If B's g3 fails again, the run stops for a human. Tested on 12 fake stops, including a copy of the real campaign's state.
- **For comparison:** A's g3 was valid on its 3rd attempt (1,087 and 1,094 words, then 938), within the protocol.

### 3b. B's g3 failed the word limit again, and one more retry (16 Nov, 17:03 GMT)

- **What happened:**
  - The first retry ran at 12:56:23. Its pass hit the 5-hour usage limit straight away, so B's g3 got no real attempt then.
  - The auto-restart waited out the limit and resumed at 16:55:42.
  - B's g3 then failed all 3 attempts again; the last had 1,072 words (17:03:43). That's 6 attempts, all between 1,019 and 1,250 words.
- **Why it matters:**
  - The runner can't write `summary.json` or `results.json` while a lineage has failed.
  - So Step 3 can't finish until B's g3 is valid, and the protocol has no rule for a rewrite that can't be written.
- **Decision:** the user approved one more retry on 16 Nov 2026, about 17:45 GMT ("Retry once more").
  - That gives B's g3 up to 9 attempts, against the protocol's 3. It's a departure, and the report must state it.
  - It can affect only P4 and lineage B's descriptive tables.
  - P1 (g1 − g0) and the check are complete: 24 + 24 and 12 searches, all from before this.
- **How:**
  - `retry_lineage_b_again.sh` is a copy of `retry_lineage_b_once.sh`, made so the running first watcher's file was never edited. Only its header and approval quote differ.
  - It was started at 17:41:39 GMT in the Terminal tab "Step 3 retry B g3 again", watching `resume_on_limit.sh` (pid 58462).
  - It reruns once, under the same conditions as the first. If B's g3 fails again, the run stops for a human.

### 3c. B's g3 was never written; B's final is g2 (16 Nov, 18:14 GMT)

- **What happened:**
  - The second retry started at 18:06:41.
  - All 3 of its attempts were also over the limit: 1,086, 1,152 and 1,173 words.
  - That's 9 attempts in all, every one over 1,000 words (1,019–1,250), and getting longer.
  - The runner stopped at 18:14:22 with only final B's 12 searches left. All 120 other meta-test searches are finished.
- **Decision (user, 16 Nov 2026, about 18:20 GMT):**
  - Treat B's g3 as not written.
  - B's final is its last kept program, g2: the protocol's own definition of final ("the last kept program").
  - Finish the run.
  - This is a departure, and the report must state it.
- **What it can affect:**
  - **P4 (final − g1):** B's contribution is g2 rather than a g3 that never existed.
  - **Lineage B's descriptive tables:** 3 generations, not 4.
  - **The ending:** not affected. P1, P2, P3 and the check use arms that are all complete.
- **How:**
  - `finish_step3_b_final_g2.py` subclasses the frozen `Step3` and overrides only `lineage()`.
  - That method is copied from the frozen source with `inspect.getsource`. The only lines that differ are marked "CHANGED": B's generation loop stops after g2, and the departure is written into `lineage.json`, `summary.json`'s notes and `progress.log`.
  - Everything else is the frozen code, unchanged: the integrity check, cached searches and calls, final B's searches, the rule review, the judge, the program review, the analysis and `results.json`.
- **Tested on a copy of the campaign, with guards that fail on any new model call or search.** Ten checks passed:
  - A's `lineage.json` is byte-identical to the real run's;
  - B's final is g2, with generations g0–g2 and the departure recorded;
  - final B is submitted with g2.
- **Integrity:** `hash_checks()` reports all 128 frozen files unchanged.
- **Unofficial interim reads, before the rule review (they don't change the run):**
  - P1 +3.1 points (raw p 0.218);
  - P2 +1.6 (0.572);
  - P3 +3.8 (0.063);
  - P4 on lineage A only −1.5 (0.656);
  - the check 8 of 12.
- **Run:** under `resume_on_limit.sh` with the finisher as its command, in a Terminal tab.
