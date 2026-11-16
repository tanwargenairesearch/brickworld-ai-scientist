# Step 3: does rewriting its own procedure help in a kind of world it has never seen? Result

**When and how:**
- **When:** 14:15:14 on 15 November 2026 to 19:58:31 on 16 November 2026: 29.7 hours elapsed, including two usage-limit stops and three stops for a human (see "Departures and disclosures").
- **Protocol:** `PROTOCOL_STEP3.md`, approved on 15 November 2026 at about 12:37 GMT ("Approve defaults"), with the code frozen in commit `01a66779`. The Step 3 manifest's sha256 is `98061d6ddb7a119c27c9d11e34660b4a5af38bb9c17bf0412b2d19a90519dad4`.
- **Approval:** `LAUNCH_APPROVED.md`, written on the user's words "Approve, launch".
- **Size:**
  - 2 lineages from the same g0. Lineage A wrote 3 rewrites; lineage B wrote 2, because its third never fit the word limit.
  - Each generation ran 7 searches, one per training world.
  - Then 132 meta-test searches on the held-out translation family (shift, rowshift, handshift), scores only: 24 each for g0, g1, final, facts-only and no-logs, plus 12 for the check (g0 with Step 2's readouts).
- **Hardware:** every training run and score ran on the laptop's Apple Silicon GPU through MPS, four at a time. There were 4,718 training runs, all ok: 1,173 in meta-training, 3,167 in the meta-test, 348 confirmations and 30 starts. The protocol estimated about 5,300.
- **Integrity:** all 128 frozen files were hashed before the first run and after the last. None changed.

**Sources:**
- Every number in the tables comes from `campaigns/step3/results.json` (status "final"). `make_report_tables3.py` prints the tables from it.
- Notes marked "descriptive" come from the run record or are computed after the run. They change no decision.

## Outcome: ending D

**Ending D:** Even with readouts, it couldn't reliably fix the held-out worlds, so this test can't say whether a better procedure would help. One model, whose weights never changed; two lineages, at toy scale; every rule re-indexes where things are, and one family was held out.

- P1 fails: g1 beat g0 by 3.1 points (Holm p 0.65).
- The check fails: 8 of its 12 searches reached 38.3%, and the protocol needed 9.
- So, by the rule fixed before the run, this test can't say whether a better procedure would help.
- "Recursive" is reserved for ending A, so this report doesn't use it for the procedure.

## Tests

Each test is a two-sided permutation test that shuffles arm labels within each world (1,000,000 shuffles), with Holm over the four. The 95% intervals are bootstrap intervals.

| Test | Arms | Difference, points | 95% interval | p | Holm p | Passes |
|---|---|---|---|---|---|---|
| P1 | g1 − g0 | +3.1 | -0.4 to +8.4 | 0.2182 | 0.6547 | **no** |
| P2 | g1 − facts | +1.6 | -3.7 to +7.4 | 0.5715 | 1.0000 | **no** |
| P3 | g1 − nologs | +3.8 | +0.4 to +8.9 | 0.0627 | 0.2507 | **no** |
| P4 (lineages A, B) | final − g1 | -0.4 | -7.0 to +6.3 | 0.8396 | 1.0000 | **no** |

Check (g0 with readouts): 8 of 12 searches at 38.3% or more; 9 needed: fails. A kept rewrite changed the header: yes.

## Per arm

"Found" counts searches whose confirmed score reached 38.3%. "Picks the start" counts searches whose final pick was the starting network, because nothing it tried beat it. The judge's verdict says whether a search's own analyses stated the hidden rule.

| Arm | Searches | Mean | By world | Found | Found by world | Picks the start | Violations | Judge yes/partly/no |
|---|---|---|---|---|---|---|---|---|
| g0 | 24 | 0.8% | handshift 0.0%, rowshift 2.0%, shift 0.4% | 0 | handshift 0, rowshift 0, shift 0 | 20 | 0 | 0/16/8 |
| g1 | 24 | 3.9% | handshift 7.5%, rowshift 0.4%, shift 3.9% | 1 | handshift 1, rowshift 0, shift 0 | 17 | 0 | 0/11/13 |
| final | 24 | 3.5% | handshift 8.7%, rowshift 1.6%, shift 0.2% | 1 | handshift 1, rowshift 0, shift 0 | 15 | 0 | 0/14/10 |
| facts | 24 | 2.4% | handshift 5.2%, rowshift 1.9%, shift 0.0% | 1 | handshift 1, rowshift 0, shift 0 | 12 | 0 | 0/10/14 |
| nologs | 24 | 0.2% | handshift 0.0%, rowshift 0.5%, shift 0.0% | 0 | handshift 0, rowshift 0, shift 0 | 23 | 0 | 0/11/13 |
| check | 12 | 60.9% | handshift 76.1%, rowshift 58.2%, shift 48.4% | 8 | handshift 3, rowshift 3, shift 2 | 0 | 0 | 12/0/0 |

## Secondary (reported only)

| Secondary | Difference, points | 95% interval | Holm p |
|---|---|---|---|
| final - g0 | +2.7 | -0.9 to +8.3 | 1.0000 |
| P1 lineage A | +6.8 | +0.4 to +16.3 | 0.2126 |
| P2 lineage A | +3.0 | -6.7 to +13.6 | 1.0000 |
| P3 lineage A | +7.6 | +1.4 to +17.1 | 0.2856 |
| P4 lineage A | -1.5 | -12.9 to +10.1 | 1.0000 |
| P1 lineage B | -0.5 | -1.6 to +0.4 | 1.0000 |
| P2 lineage B | +0.1 | -0.3 to +0.7 | 1.0000 |
| P3 lineage B | -0.0 | -0.6 to +0.6 | 1.0000 |
| P4 lineage B | +0.7 | -0.6 to +2.6 | 1.0000 |

Lineage A's g1 beat g0 by 6.8 points (raw p 0.024), and lineage B's by −0.5. That spread between the two lineages is why each rewritten arm is best read as two programs, not as rewriting in general.

## Meta-training

These are the training worlds, scored by the same searches whose logs each rewrite learned from. They show the loop working as designed, not transfer.

| Lineage | Generation | Program | Header (R, K, seeds) | Mean confirmed | Per search | Kept | Parent header used |
|---|---|---|---|---|---|---|---|
| A | 0 | g0 | 4, 3, 2 | 0.0% | 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 | start |  |
| A | 1 | g1 | 4, 6, 1 | 24.9% | 0.796, 0.000, 0.000, 0.031, 0.000, 0.918, 0.000 | yes |  |
| A | 2 | g2 | 4, 6, 1 | 78.9% | 0.782, 0.779, 0.939, 0.842, 0.863, 0.884, 0.430 | yes |  |
| A | 3 | g3 | 3, 4, 2 | 82.0% | 0.967, 0.750, 0.937, 0.920, 0.784, 0.900, 0.482 | yes |  |
| A | final | g3 | | | | | |
| B | 0 | g0 | 4, 3, 2 | 2.4% | 0.000, 0.167, 0.000, 0.000, 0.000, 0.000, 0.000 | start |  |
| B | 1 | g1 | 4, 6, 1 | 41.3% | 0.118, 0.763, 0.116, 0.212, 0.141, 0.770, 0.771 | yes |  |
| B | 2 | g2 | 4, 6, 1 | 76.5% | 0.874, 0.775, 0.891, 0.349, 0.695, 0.878, 0.893 | yes |  |
| B | final | g2 | | | | | |

## Programs

The word counts use the protocol's fixed list. "roll" also matches "rollout", so most of its counts are recipe settings, not hints.

| Program | Words | Kinds of change named (counts) | Program review |
|---|---|---|---|
| A/facts | 675 | flip 1, transpose 1, permute 1 | no rule-breaking advice |
| A/g0 | 344 | none | no rule-breaking advice |
| A/g1 | 989 | roll 1, flip 1, permute 1 | no rule-breaking advice |
| A/g2 | 942 | shift 1, roll 1, flip 2, transpose 6, permute 2, remap 2 | no rule-breaking advice |
| A/g3 | 938 | roll 2, transpose 4, permute 2, remap 1 | no rule-breaking advice |
| A/nologs | 929 | turn 1 | no rule-breaking advice |
| B/facts | 677 | flip 2, transpose 2, remap 1 | no rule-breaking advice |
| B/g0 | 344 | none | no rule-breaking advice |
| B/g1 | 842 | roll 1, flip 4, transpose 3, permute 1, remap 1, symmetr 1 | no rule-breaking advice |
| B/g2 | 936 | roll 1, flip 11, transpose 7, symmetr 4 | no rule-breaking advice |
| B/nologs | 906 | roll 2 | no rule-breaking advice |

**Family-level hints, quoted as the protocol requires.** Two kept programs mention moves that a translation fix would use:
- Lineage A's g2, under "Later rounds": "try products of hypotheses … and new learned remaps, such as one-cell anchor shifts or learned input mixing in case channel roles are swapped."
- Lineage A's g3 lists them among its "Failed families (do not spend slots on them)": "… logit temperature; index rolls; 5x5 windows; …".

So the final program of lineage A told its searches to avoid index rolls, the kind of change that fixes the held-out family. That's what fitting to the training worlds looks like: every training rule is a symmetry of the square, and none is a roll.

## Calls and cost

| Calls | Count | Notional cost |
|---|---|---|
| judge | 132 | $3.67 |
| program_review | 10 | $0.49 |
| review | 45 | $2.21 |
| rewrite | 9 | $6.17 |
| search: check | 49 | $18.77 |
| search: facts | 107 | $37.74 |
| search: final | 87 | $63.21 |
| search: g0 | 107 | $35.53 |
| search: g1 | 120 | $101.39 |
| search: meta | 214 | $118.45 |
| search: nologs | 110 | $36.44 |
| total | 990 | $424.08 |

The total is notional: the calls ran on a subscription. It's above the protocol's estimate of $250–350. Failed rewrite attempts aren't in the count.

## Also recorded (descriptive, computed after the run; no decisions)

- **The check missed by one search.**
  - 8 of 12 is consistent with a true rate anywhere from 35% to 90% (Clopper–Pearson 95% interval).
  - Even at a true rate of 75%, a 12-search check with a pass mark of 9 fails 35% of the time.
- **The check's searches found the rule but didn't always turn it into a fix.**
  - The judge said yes in all 12 check searches: their analyses stated the hidden rule.
  - Yet only 8 reached 38.3%.
  - By world: handshift 3 of 4, rowshift 3 of 4, shift 2 of 4.
- **No large gain appeared.**
  - Before the run, P1 had about 91% power to detect fixes in 40% of searches against about 0%.
  - g1 found a fix in 1 of 24 searches, and g0 in none.
- **No arm learned to fix the held-out family from scores.**
  - Every scores-only arm found a fix in at most 1 of its 24 searches, always in handshift.
  - Scores-only searches mostly kept the starting network: from 12 of 24 (facts-only) to 23 of 24 (no-logs).
- **Rule review:** every program passed: "no rule-breaking advice".
- **Checks:** 45 picks were reviewed for compliance, and all passed.

## What this can and can't show

- **It can't say whether rewriting helps.** The test was too weak to tell: the held-out family was hard to fix even with readouts. That's what ending D says.
- **The training-world gains are in-sample.** They rose from 0.0% to 82.0% (A) and from 2.4% to 76.5% (B). The same worlds supplied the logs, scored each rewrite and decided what was kept.
- **Two programs per arm.** The lineages differed a lot, so these tests are about two programs, not about rewriting in general.
- **P2 mixes the header with the text.** Both first rewrites changed the header to 4 × 6 × 1, and facts-only keeps g0's header.
- **One held-out family, toy scale, one model whose weights never change.** At most, a test like this could show a research procedure improving itself, not a model.
- **Lesson for next time:** run the check before the main test, so a too-hard held-out family is caught before it costs a full run.

## Departures and disclosures

The details and times are in `RUN_NOTES.md`.

1. **Usage limits.** On 15 November at about 21:11, the subscription's usage limit made calls fail.
   - The frozen call code didn't recognise that message as an infrastructure error, so the runner stopped instead of waiting.
   - The user approved an automatic restart, `resume_on_limit.sh` ("Yes, auto-restart"). It reruns the same command and reuses every finished call and training run.
   - The limit stopped the run twice in all (15 and 16 November). No saved prompt or reply contains the limit message.
2. **Lineage A's g2 took 5 attempts**, against the protocol's 3. Its first 3 attempts and the next one were over 1,000 words, and the fifth was valid at 942 words.
3. **Lineage B's g3 was never written.**
   - All 9 attempts were over 1,000 words (1,019–1,250), across three runs approved by the user ("Yes, retry automatically", then "Retry once more").
   - On 16 November at about 18:20 GMT the user decided: treat B's g3 as not written, and make B's final its last kept program, g2, the protocol's own definition of final.
   - `finish_step3_b_final_g2.py` finished the run. It subclasses the frozen runner and overrides one method, copied from the frozen source with only the marked lines changed, so the frozen files are untouched but that one case behaves differently.
   - It was tested on a copy of the campaign, and all 128 hashes were unchanged.
4. **Interim numbers were seen before that decision.** Unofficial reads of P1–P3, the check and lineage A's P4 were computed at about 16:10 GMT, from the run record and before the rule review.
5. **What the departures can affect.** None of them can change P1, P2, P3 or the check: those use g0, g1, facts-only, no-logs and the check arm, all finished before any departure. Departures 2–4 can affect only P4 and the descriptive tables.
