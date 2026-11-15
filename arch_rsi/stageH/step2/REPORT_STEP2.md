# Step 2: does the AI scientist find hidden rules from evidence? Result

**When and how:**
- **When:** 17:16:56 on 14 November 2026 to 07:17:47 on 15 November 2026 (14.01 h).
- **Protocol:** `PROTOCOL_STEP2.md`, with its "Changes made while building" and "Changes after the dry run" sections, all written before the full run. The code was frozen in commit `3165e149`. The Step 2 manifest's sha256 is `ae2a051f397337072370200c6ba9b7875d0a5959dd39d59cf5e908a53d0981d9`.
- **Approval:** `LAUNCH_APPROVED.md`, written on the user's words "Fix judge, then run".
- **Size:** 2 worlds (shifted booklet, mirrored hands) × 4 arms × 12 searches = 96 searches. Each search had 4 rounds of 3 candidates, each trained on 2 seeds. Each pick was then retrained on 4 new seeds (5001–5004).
- **Hardware:** every training run and score ran on the Mac's GPU, four at a time. There were 2,522 training runs.

**Sources:**
- Every number in the tables comes from `campaigns/step2/results.json` (status "final"). They are printed from it by `make_report2_tables.py`.
- The notes marked "descriptive, from the run record" come from `summary.json`, the searches' `state.json` files and `progress.log`. They don't affect any decision.
- **A score** is a pick's exact-build success on the 512 dev designs, as the mean over seeds 5001–5004.
- **"Found"** means a score of 38.3% or more.

**Integrity:**
- Stage 2b's 39 frozen hashes, Stage H's 26 and Step 2's 12 were unchanged before the first run and after the analysis.
- Step 2 has no holdout, and no holdout or final file was read.
- Stage 2e stayed paused.
- The dry run (`campaigns/step2_dry`) is excluded from everything below.

**What ran:**
- **Search calls:** all 96 searches finished. 406 calls: 384 rounds plus 22 repair calls, none of which failed.
- **Trials:** 1,152. 1,151 ran "ok". One was refused by the check before training, because its recipe allowed too few updates.
- **Review:** all 52 distinct picks that weren't the start were reviewed blind, and all were compliant.
- **Judge:** all 96 searches were judged.

## Outcome: P1 and P2 pass, and the check passes

> With readouts of its own runs, the AI scientist found fixes for hidden rules it wasn't told, more often than with scores only or with none. Two rules of one kind, toy scale, one model, whose weights never changed.

This is the protocol's pre-set ending for "P1 and P2 pass", word for word.

- **P1, readouts − scores hidden:** +75.6 points. None of 1,000,000 within-world shuffles was as large, so p is at the test's floor, and Holm p is 0.000002.
- **P2, readouts − scores only:** +75.9 points, with the same p.
- **The check, a precondition for reading P1 and P2 as discovery:** the told arm found a fix in 19 of 24 searches. It needed 18 (75%), so it passes.
- **Counts:** readouts found a fix in 23 of 24 searches, against 0 of 24 for scores only and 0 of 24 for scores hidden.

## Per arm

| Arm | Found (all) | Shift: found | Mirror: found | Mean score | Shift mean | Mirror mean | Picks = start | Violations | Judge yes / partly / no |
|---|---|---|---|---|---|---|---|---|---|
| Told the rule | 19/24 | 7/12 | 12/12 | 64.9% | 43.6% | 86.2% | 0 | 0 | 24 / 0 / 0 |
| Readouts | 23/24 | 11/12 | 12/12 | 76.2% | 67.7% | 84.7% | 0 | 0 | 24 / 0 / 0 |
| Scores only | 0/24 | 0/12 | 0/12 | 0.3% | 0.0% | 0.6% | 23 | 0 | 0 / 16 / 8 |
| Scores hidden | 0/24 | 0/12 | 0/12 | 0.7% | 1.3% | 0.0% | 21 | 0 | 0 / 0 / 24 |

- **Picks = start:** searches whose final pick was the starting architecture, because nothing beat its 2-seed score of 0.
- **Judge:** a blind call per search, given the rule and every round's analysis, which said whether the text states the rule. It is descriptive only.

## Tests

| Test | Arms | Difference in mean score | 95% bootstrap interval | Permutation p | Holm p | Passes |
|---|---|---|---|---|---|---|
| P1 | Readouts − scores hidden | +75.6 points | +69.1 to +81.8 | < 0.000001 | 0.000002 | yes |
| P2 | Readouts − scores only | +75.9 points | +69.4 to +82.0 | < 0.000001 | 0.000002 | yes |
| S1 | Told the rule − readouts | -11.3 points | -22.3 to -0.5 | 0.071 | 0.142 | (secondary, no pass rule) |
| S2 | Scores only − scores hidden | -0.3 points | -1.5 to +0.6 | 0.739 | 0.739 | (secondary, no pass rule) |

- **P1 and P2:** pre-set, Holm over the two.
- **S1 and S2:** secondary, with no pass rule.
- **S1's two measures disagree:** its bootstrap interval just excludes 0, but its permutation p doesn't reach 0.05. Neither is a pass rule, so S1 is read as "not shown".

## Every search's score

| Arm and world | Confirmed score of each search's pick (search 0 to 11) |
|---|---|
| Told the rule, shift | 1.9, 17.2, 54.6, 27.3, 87.4, 74.4, 1.9, 0.0, 48.5, 66.4, 58.1, 85.8 |
| Readouts, shift | 55.0, 90.1, 77.9, 71.1, 55.0, 95.0, 89.6, 23.5, 81.9, 49.0, 65.8, 58.6 |
| Scores only, shift | 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 |
| Scores hidden, shift | 0.0, 0.0, 11.3, 4.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 |
| Told the rule, mirror | 87.1, 79.1, 84.9, 80.5, 94.0, 74.0, 86.7, 92.8, 89.9, 90.3, 85.6, 89.6 |
| Readouts, mirror | 91.9, 89.7, 94.2, 71.0, 91.4, 65.7, 91.8, 67.4, 89.7, 92.5, 92.8, 78.5 |
| Scores only, mirror | 0.0, 0.0, 7.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 |
| Scores hidden, mirror | 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 |

## Calls and cost

| Calls | Number | Notional cost |
|---|---|---|
| Review | 52 | $2.36 |
| Judge | 96 | $2.40 |
| Told the rule | 99 | $37.47 |
| Readouts | 97 | $35.86 |
| Scores only | 102 | $35.47 |
| Scores hidden | 108 | $30.02 |
| **All** | **554** | **$143.58** |

- **Total:** 554 calls, $143.58 notional, as the CLI reports it.
- **Estimates:** the dry run predicted about $141 (`PROTOCOL_STEP2.md`, "Changes after the dry run"). The first estimate was $90–130.

## Also recorded (descriptive, from the run record; no decisions)

1. **The told arm's 5 misses were all in the shift world.**
   - In 4 of them a trial had reached 45–88% on its 2 trial seeds. The pick then scored 0–19% on most of the 4 new seeds.
   - The fifth never passed 19.3% on its 2 seeds.
   - These searches' own analyses name the rule and blame collapse on some seeds, to early DONE or to stopping after layer 0. One says: "Most of the gap between runs is seed collapse into early DONE, not skill".
   - Their analyses suggest the misses were unreliable training, not a failure to write the fix. The confirmation on 4 new seeds caught them.
2. **The readouts arm in the shift world:** it did better than the told arm there (11 of 12 against 7 of 12). Both arms see the same readouts, so the readouts alone don't explain the gap; the only difference is the rule text. S1 doesn't show a difference, so this is an observation, not a result.
3. **Scores only against scores hidden:** the judge found the scores-only arm naming the right kind of change ("partly") in 16 of 24 searches, and the scores-hidden arm in none. With scores stuck at 0 it guessed the world might remap actions or observations, but it never turned that into a network that scored. Both arms found nothing (S2: −0.3 points).
4. **Partial scores without the rule:** three picks scored above 0 without the rule: 11.3% and 4.2% (scores hidden, shift) and 7.7% (scores only, mirror). All three are named for wider or whole-board context ("dilated context conv", "global context", "whole-board context"), and none reached 38.3%.
5. **How early fixes were found:** this counts the round in which a trial first reached 38.3% on its 2 seeds (`first_round_found` in `results.json`).
   - Readouts: round 1 in 19 of 24 searches and round 2 in the other 5.
   - Told the rule: round 1 in 16, round 2 in 5, round 3 in 1 and round 4 in 1, with 1 that never did.

## What this can and can't show

- **It can show:** given readouts of its own runs, this AI scientist read off two hidden rules it wasn't told and wrote working fixes. That happened in 23 of 24 searches. With only the score, or with nothing, it never did.
- **It can't show:**
  - **Other kinds of rule:** both rules are of one kind, a spatial remapping, and Stage H1's two other rules weren't usable.
  - **That it would choose what to look at:** we designed the readouts, knowing the rules. A test checked before the run that each rule's evidence shows: in the shift world the observed target height map is the design rolled 4 columns, and in the mirror world a brick chosen at column x fills columns 8 − dx − x to 7 − x (`test_prompts_step2.py`, `readouts_show_each_rules_evidence`). The AI scientist read them; it didn't choose them.
  - **Larger scale, other models, or that the loop improves itself:** this is one model, whose weights never changed, at toy scale. The loop itself was not changed.
- **For Step 3:** the protocol says a P1 pass means the evidence-reading part of the loop works, so Step 3 would test whether the loop can improve its own procedure. Step 3 still needs at least three usable rule families, and Stage H1 found two (`PROTOCOL_STEP2.md`, "Step 3").

## Departures and disclosures

1. **Judge change:** the judge was changed after the dry run and before the full run, to read every round's analysis. It's disclosed in the protocol and descriptive only.
2. **Cost:** $143.58 notional, against the first estimate of $90–130. That's within $3 of the dry run's estimate.
3. **One invalid trial:** shift / scores hidden / search 3 / r4c2. Its short check run failed with "the budget allows only 3 updates of 1024 steps; at least 4 are required". It was not trained, and it counts as a failed trial, as the rules say.
4. **Repair calls:** 22 (told 3, readouts 1, scores only 6, scores hidden 12). Two rounds needed a second repair, and none needed more than the 2 allowed.
