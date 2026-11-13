# Stage 1 addendum: how the three outcomes are read

Written on 9 November 2026 at about 09:45 UTC, while Stage 1 was still running.

**What existed when this was written:**
- `STAGE1_DONE` was not in the log.
- `confirm/summary.json`, `review.json` and `results.json` did not exist.
- `analyze_stage1.py` had not been run on the live campaign.

**What had already been seen**, in passing, in `run.log` and `progress.log`:
- the two comparators' confirmation means, both about 22%;
- round-by-round selection scores of some searches;
- one menu search's confirmation seeds.

No g0 or no-feedback confirmation result had been seen.

This file does not change `protocol_stage1.md`, which is hash-checked and stays as written. It only fixes the meaning of one undefined phrase, "g0 ≈ nofb", before any result that the phrase concerns exists. Its SHA-256 is recorded in the hand-over note.

## Definitions

All outcomes are read from `analyze_stage1.py`'s output with `primary_family.status` equal to "final". That means after `review_stage1.py`, on the compliant (primary) scores, never the as-computed ones.

- **P1** is `g0 - nofb` and **P2** is `g0 - start`. Both are in the primary family, Holm-adjusted over 2 contrasts.
- **S3** is `nofb - start` in the secondary family, Holm-adjusted over 6 contrasts.
- A contrast **"passes"** when its difference is above 0 and its Holm-adjusted p is below 0.05.

## Outcomes

**A. The gate passes:** P1 and P2 both pass.
- Allowed sentence: "experiments guided the search".

**B. "g0 ≈ nofb, and both beat the start":** P1 does not pass (any sign, any p), P2 passes, and S3 passes.
- This is a gate failure. It goes back to the user before Stage 2.
- Allowed sentence: "Claude's first ideas plus selection beat the start; experiments added nothing measurable".
- "≈" means "no difference was shown", not "they were shown to be equal". There is no equivalence test. So every report of B states the estimated power from `protocol_stage1.md`: 40–62% for a true +15 points.

**C. The gate fails, any other result.** For example:
- P2 does not pass;
- P1 does not pass, P2 passes, but S3 does not pass;
- P1 passes and P2 does not.

What may be written:
- No verdict sentence beyond "the gate failed".
- Each contrast may be reported as its numbers (difference, Holm-adjusted p, bootstrap interval) without a verdict word.
- It goes back to the user before Stage 2.

**Special case: P1 shows the opposite direction** (difference below 0 with Holm-adjusted p below 0.05: hiding the scores did better). This is outcome C, reported as numbers. No explanation is offered as a finding.

## Always

- If the rule review changes the outcome, the as-computed outcome is stated beside the primary one.
- The final split stays locked.
