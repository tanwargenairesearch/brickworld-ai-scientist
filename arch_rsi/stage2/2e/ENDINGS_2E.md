# Stage 2e endings (written before any Stage 2e run)

Pre-written text for the article once Stage 2e finishes. Exactly one ending is used.

- **Written:** 10 November 2026, before any Stage 2e training run or LLM call, including the dry run.
- **Numbers:** nothing here contains a 2e number. Every `[2E: …]` slot is filled from the campaign's `results.json`, using the key table at the end.
- **Reading rules:** from `PROTOCOL_2E.md` ("What it can and can't show" and "Analysis").
- **Wording:** generic, as in the article. No model names and no person names. A null result is said as what the test did or didn't show, never as a failure. The sentences below may be edited for style; what they claim may not change: the reading, its numbers and the limits.

## Before choosing an ending

1. `STAGE2E_DONE` is in `campaigns/stage2e/progress.log`.
2. `review_2e.py` has run after the last pair, and `review.json` has a verdict for every pick.
3. `analyze_2e.py` has run after that, and `results.json` has `status` equal to `"final"`. If it says `"pending review"`, stop: no ending applies.
4. Every `[2E: …]` comes from that `results.json`. Never from `progress.log`, the round-by-round selection scores or any interim number.
5. W's final split stays locked: `stage2/FINAL_OPENED_W` must not exist.

## Which ending

`primary.reading` in `results.json` names it. The rule, from the protocol:

| Ending | `primary.reading` | When |
|---|---|---|
| **P** | `pass` | the mean difference is positive and p < 0.05 |
| **B** | `bounded` | no pass, and the interval's upper end is below +10 points |
| **O** | `open` | no pass, and the upper end is +10 points or more |
| **W** | `worse` | the mean difference is negative and p < 0.05 |

Read the primary scores only, never the as-computed ones.

## The shared opening (every ending)

> **2e. A longer search, tested a third time.** Each pair of searches shared its first three candidates, written with no scores to see. Then the pair split. One branch saw every score from then on, and the other saw none, for seven more rounds each: 24 tries per search, against 12 before. Both branches picked with the same rule on the real scores. I ran [2E: n_pairs] pairs and confirmed each pick on 16 fresh seeds:
> - seeing its scores: [2E: seen mean]
> - scores hidden: [2E: hidden mean]
> - for comparison, the start on the same 16 seeds: [2E: start mean]

## Ending P (pass)

> The difference was +[2E: d] points (interval [2E: lo] to [2E: hi], p [2E: p]), on the one test fixed before the run. With twice the tries, experiments guided the search.
>
> That is one of three tests. At 12 tries, in Stage 1 and in Stage 2b, the same comparison came out at +1.5 points, with no measurable benefit. I report all three.

- **Ladder, row 1b:** "Shown at 24 tries (Stage 2e). Not shown at 12 tries (Stages 1 and 2b)."
- **Stage 2c:** eligible again. It needs its own approval, and its protocol's gate sentence needs an amendment first, because it names 2b's gate. Say only that it is the next step if approved.
- **Subtitle:** "Twice, the AI scientist's searches did as well with their results hidden. With twice the tries, the third test came out the other way."

## Ending B (bounded)

> The difference was [2E: d signed] points (interval [2E: lo] to [2E: hi], p [2E: p]). The test didn't show that seeing its scores helps. The interval's upper end is below the 10 points I set in advance, so at 24 tries any benefit from seeing its scores is under [2E: hi] points.
>
> Three tests now agree. That still doesn't show feedback never helps: a smaller benefit, or one at a larger budget, stays possible.

- **Ladder, row 1b:** "Not shown, three times. At 24 tries, any benefit is under [2E: hi] points."
- **Stage 2c:** doesn't run.
- **Subtitle:** "Three times now, the AI scientist's searches did as well with their results hidden."

## Ending O (open)

> The difference was [2E: d signed] points (interval [2E: lo] to [2E: hi], p [2E: p]). The test didn't show that seeing its scores helps. The interval still reaches [2E: hi] points, above the 10 I set in advance, so this test can't rule out a benefit that large.

- **Ladder, row 1b:** "Not shown, three times. The third test couldn't rule out a benefit of up to [2E: hi] points."
- **Stage 2c:** doesn't run.
- **Subtitle:** as in ending B.

## Ending W (worse)

> Seeing its scores made the picks worse, by [2E: −d] points (interval [2E: lo] to [2E: hi], p [2E: p]). I report it and don't explain it: the test wasn't built to say why.

- **Ladder, row 1b:** "Not shown. At 24 tries, seeing its scores made picks worse."
- **Stage 2c:** doesn't run.
- **Subtitle:** "Twice, hiding the results cost nothing. With a longer search, showing them did worse."

## Extra rules for every ending

- **The number of pairs.** Say how it was set, in one sentence: "The first 32 differences had a spread of [2E: rms], [above | at or below] the 0.20 I'd fixed, so [16 more pairs ran | the run ended at 32]. That rule never reads which branch won."
- **If a ceiling stopped the run** (`ceiling_stop` isn't null), say so, and that the analysis uses the [2E: n_pairs] pairs that finished.
- **If any pick was judged to break a rule** (`violations` > 0 in either branch), say how many and that each was scored as the start.
- **Pairs with the same pick:** report `same_pick_pairs` as a number. Those pairs differ by exactly 0.
- **Reported only, never claimed:** each branch against the start; the running best by round; selection optimism.
- **Always state the limits:**
  - one toy world, one model;
  - dev designs, used before by Stage 2b's searches, with held-out designs still locked;
  - the AI scientist's weights and research program unchanged;
  - the rule review is blind, not independent.

## Key table

| Slot | `results.json` |
|---|---|
| n_pairs | `primary.n_pairs` |
| d, d signed, −d | `primary.mean_d` × 100, one decimal |
| lo, hi | `primary.ci95_boot` × 100, one decimal |
| p | `primary.p`, two significant figures; "below 0.001" under that |
| seen mean, hidden mean | `per_branch.seen.mean`, `per_branch.hidden.mean`, as percentages |
| start mean | `start.mean`, as a percentage |
| rms | `second_stage.rms`, two decimals; `second_stage.extend` says which wording |
| violations | `per_branch.seen.violations`, `per_branch.hidden.violations` |
| same_pick_pairs | `same_pick_pairs` |
