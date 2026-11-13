# Stage 2c endings, written before any result

Written 10 November 2026, about 13:45 GMT. At that time Stage 2b was still running, 2c hadn't been calibrated or run, and no 2c result existed. The video's scene 4 (`arch_rsi/viz/stage2c/make_2c_video.py`) and the article use exactly one of these endings. Which one is set mechanically from `results.json` (`analyze_2c.py`), by the protocol's tiers:

| Ending | When |
|---|---|
| `recursive` | `tiers.words_allowed` is "recursive": P1, P2, P3 and P4 all positive with Holm-adjusted p < 0.05, and a kept rewrite changed the header |
| `level2` | `tiers.words_allowed` is "Level 2": P1, P2 and P3 all positive with Holm-adjusted p < 0.05 |
| `transfer_only` | Neither, but P1 is positive with Holm-adjusted p < 0.05 |
| `none` | Anything else |

**Rules for every ending:**
- Numbers are filled in from `results.json` only: the {placeholders} below. {P1} to {P4} are differences in points (signed, one decimal); {P1p} to {P4p} are Holm-adjusted p-values; {P4txt} is "{P4} points, Holm-adjusted p {P4p}", or "not tested: no lineage kept a rewrite".
- Each ending is worded to be true in every case that selects it.
- Generic wording: "the AI scientist", with no model or person names.
- Only the `recursive` ending claims "recursive". The `level2` ending may say that its bar wasn't met.

## recursive
video: Rewriting its own procedure helped on a larger world, beyond added facts and beyond rewriting with shuffled scores, and kept rewrites added more ({P1} points for the first rewrite, {P4} more by the last kept one).
article: On the larger world the AI scientist never saw while rewriting, its first rewritten program beat the original by {P1} points. It also beat a rewrite that could only add facts ({P2}) and a rewrite from shuffled scores ({P3}). The last kept program added {P4} points more. All four held after correcting for four comparisons, and at least one kept rewrite changed how the search runs, not just what it advises. That is the narrow, measured sense in which this loop was recursive: two lineages, one toy world, one model whose weights never changed.

## level2
video: One rewrite of its own procedure helped on a larger world ({P1} points), beyond added facts and beyond rewriting with shuffled scores. The bar for "recursive" wasn't met.
article: On the larger world the AI scientist never saw while rewriting, its first rewritten program beat the original by {P1} points. It also beat a rewrite that could only add facts ({P2}) and a rewrite from shuffled scores ({P3}), each after correcting for four comparisons. So the procedure change, made from real evidence, helped. The bar for calling it recursive wasn't met: that needs the last kept program to beat the first rewrite (here: {P4txt}) and a kept rewrite that changed how the search runs, both together. One self-revision helped; I can't call it recursive.

## transfer_only
video: The rewritten program beat the original on a larger world ({P1} points), but didn't beat both controls, so the gain can't be credited to rewriting the procedure from evidence.
article: The first rewritten program did better than the original on the larger world ({P1} points, after correcting for four comparisons). But it didn't beat both controls: a rewrite that could only add facts ({P2}) and a rewrite from shuffled scores ({P3}). The gain might come from carried-over knowledge, or from rewriting at all, rather than from using the evidence to change the procedure. That's a weaker result than self-improvement, and I report it as that.

## none
video: No measurable gain from the AI scientist rewriting its own procedure at this scale ({P1} points for the first rewrite). With 16 searches per arm, gains under about 15 points are hard to detect.
article: Rewriting its own research program didn't measurably help on the larger world. The first rewrite changed the score by {P1} points against the original (Holm-adjusted p {P1p}); against the controls, by {P2} (facts-only, p {P2p}) and {P3} (shuffled scores, p {P3p}). With 16 searches per arm, the design detects a 15-point gain about 4 times in 5 and a 5-point gain rarely, so "not established" is the honest reading, not "it never helps".
