# Step 3a: more hidden rules for Step 3. Result

**When and how:**
- **When:** 07:22:12 to 08:08:28 GMT on 15 November 2026 (46.3 min from the tests to the end of the analysis). The protocol's estimate was about 1–1.5 h.
- **Protocol:** `PROTOCOL_STEP3A.md`, approved on 14 November 2026 at 18:58 GMT, with its "Changes made while building" section, all written before any run. The code was frozen in commit `c3f3a9c1`. The Step 3a manifest's sha256 is `c73e8fbef7381b18f096be41edeed15459345fa5ca40312e0e84a4027850ee6d`.
- **Launch:** `wait_then_run_3a.sh` (commit `79b9b155`), on the user's words "Yes, launch then". It waited until Step 2's results file existed and no Step 2 runner was alive, then started `run_3a.py`.
- **Size:** 150 runs: the shared plain cell, plus 8 rules × 3 cells, each on seeds 4000–4005. Every run was 131,072 steps on the Mac's GPU, four at a time.

**Sources:**
- Every number below comes from `runs/results.json` (status "final"). `make_report_tables_3a.py` prints the tables from it.
- **A score** is a run's exact-build success on the 512 H-plain dev designs.

**Integrity:**
- Stage 2b's 39 frozen hashes, Stage H's 26, Step 2's 12, Step 2S's 12 and Step 3a's 22 were unchanged before the first run and after the analysis.
- No holdout or final file was read: these worlds have none.
- Stage 2e stayed paused. No model was called.

**What ran:**
- **Tests:** `test_rules_3a` (CPU rules, 5 s) and `test_identity_3a` (GPU, 175 s) both passed before the first run.
- **Legality checks:** all 8 ran and passed.
- **Runs:** all 150 finished "ok" on their first attempt. None failed or timed out, so no run was repeated.
- **Fallback:** not triggered. Every rule passed at 131,072 steps.

## Outcome: all 8 new rules pass, so Step 3 can hold out a whole family

- **New rules passing:** 8 of 8: H5–H12.
- **Usable families** (at least two passing members, counting H1 and H2): all four, translation, reflection, transposition and rotation.
- **Decision:** the protocol's case for 3 or 4 usable families applies: **Step 3's protocol can hold out a whole family.** It is written after Step 2's result, which is final, and doesn't need Step 2S, which now runs after Step 3.

## Per rule

| Rule | Family | Plain, ref A | Plain, fix | Rule, ref A | Rule, fix | Repair, points (95% interval) | Holm p | Rule-specific gain, points | Recovery | Passes |
|---|---|---|---|---|---|---|---|---|---|---|
| H5, booklet shifted 4 rows | translation | 76.6% | 0.0% | 0.0% | 76.6% | +76.6 (+72.5 to +80.4) | 0.0173 | +153.2 | 1.00 | **yes** |
| H6, hands shifted 4 columns | translation | 76.6% | 0.0% | 0.0% | 74.1% | +74.1 (+69.0 to +77.9) | 0.0173 | +150.7 | 0.97 | **yes** |
| H7, hands mirrored top to bottom | reflection | 76.6% | 0.0% | 0.0% | 71.1% | +71.1 (+65.2 to +78.3) | 0.0173 | +147.7 | 0.93 | **yes** |
| H8, booklet mirrored left to right | reflection | 76.6% | 0.0% | 0.0% | 76.6% | +76.6 (+72.5 to +80.4) | 0.0173 | +153.2 | 1.00 | **yes** |
| H9, booklet transposed | transposition | 76.6% | 0.0% | 0.0% | 76.6% | +76.6 (+72.5 to +80.4) | 0.0173 | +153.2 | 1.00 | **yes** |
| H10, booklet transposed on the other diagonal | transposition | 76.6% | 0.0% | 0.0% | 76.6% | +76.6 (+72.5 to +80.4) | 0.0173 | +153.2 | 1.00 | **yes** |
| H11, booklet turned a quarter turn | rotation | 76.6% | 0.0% | 0.0% | 76.6% | +76.6 (+72.5 to +80.4) | 0.0173 | +153.2 | 1.00 | **yes** |
| H12, booklet turned three quarters | rotation | 76.6% | 0.0% | 0.0% | 76.6% | +76.6 (+72.5 to +80.4) | 0.0173 | +153.2 | 1.00 | **yes** |

- **Damage:** every rule took reference A from 76.6% to 0.0%, 76.6 points.
- **p:** every rule's 6-against-6 permutation p is 0.0022, the smallest that test can give (2/924). After Holm across the eight, each is 0.0173.
- **Every pass rule held for every rule:** repair at least 25 points with Holm p below 0.05, a rule-specific gain at least 20 points, legal, and complete.

| Family | Members | Passing | Usable for Step 3 |
|---|---|---|---|
| Translation | H1, H5, H6 | H1, H5, H6 | yes |
| Reflection | H2, H7, H8 | H2, H7, H8 | yes |
| Transposition | H9, H10 | H9, H10 | yes |
| Rotation | H11, H12 | H11, H12 | yes |

## Legality checks

Each fixed network was checked in its rule world, on the 256 check designs, on the GPU.

| Rule | Untrained: exact | Untrained: clean fill | After the short run: exact | Passed |
|---|---|---|---|---|
| H5, H6, H8, H9, H10, H11, H12 | 0.0% | 2.1% | 0.0% | yes |
| H7 | 0.0% | 2.0% | 0.0% | yes |

The limits are at most 2%, 10% and 25%. Each short run made 7 updates.

## Identity checks

- **The plain cell against Step 1's:** all six seeds reproduced Step 1's scores, design by design (73.0%, 83.2%, 80.9%, 78.9%, 75.0%, 68.6%).
- **Each booklet rule's fixed cell against the plain cell:** identical design by design for all six (H5, H8, H9, H10, H11, H12), as the protocol predicted. Their fix gives the network exactly the plain input.

## Also recorded (descriptive, no decisions)

- **Hands rules against booklet rules:**
  - The two hands rules, H6 and H7, recovered 97% and 93% of the damage, not 100%. Their fixes undo the rule in distribution, not bit for bit, so training takes a different path.
  - H6's seeds ran 62.7% to 79.3%; H7's ran 60.9% to 87.1%. The plain cell's ran 68.6% to 83.2%.
- **Designs with hidden cells:** the plain cell and every booklet rule's fixed cell built 72.3% of them exactly; H6's fix 69.3%; H7's 65.9%. Reference A in every rule world built 0.0%.
- **Wrong cells:** in every rule world, reference A's wrong cells spread over all rows and columns, with fewer at the edges, as in the plain world's pattern but about 20 times as many. With the fix they return to the plain world's counts (exactly, for the booklet rules).
- **Learning curves:** at the last logged update (122,880 steps), the plain cell and every booklet rule's fixed cell trained to 65.3% exact on training episodes; H6's fix 63.7%; H7's 59.8%. Every rule world with reference A, and every plain world with a fix, stayed at 0.0%.

## What this can and can't show

| It shows | It doesn't show |
|---|---|
| Each of the 8 rules wipes out reference A, and one known, legal change brings it back | That the AI scientist can find that change (that is Step 3's inner search, as it was Step 2's) |
| The gain comes from the rule, not from a better network: each fix scores 0.0% in the plain world | That each fix is the best possible one |
| Step 3 has four families to hold out from | That transfer across them tests procedure rather than a shared fact |

**For Step 3's protocol:**
- **All the same kind:** like H1 and H2, all ten passing rules are re-indexings of where things are. A procedure such as "check whether the input is re-indexed" could carry over as a fact about this set of worlds. Step 3's facts-only control is meant to test that.
- **Exact undo in six worlds:** in the six booklet worlds, the fix makes training exactly the plain world's. A search there that finds the exact undo scores exactly what the plain world gives on the same seeds; without it, reference A scores 0.0%. No partial fix was run, so how partial fixes score is unknown.
- **The families overlap as transforms:** a quarter turn is a transpose followed by a flip, and H1 and H6 both roll. Step 3's protocol has to choose its held-out family with this in mind.
- **Scale:** toy scale, one world size, one base network.

## Departures and disclosures

- **Another job on the GPU, briefly:** at about 07:47, while four of the main runs were training (runs 74–76 of 150 finished then), the phone video's render ran on the same GPU for about 11 s, replaying five saved Stage H1 runs. It changed no file here.
  - Every identity check on runs from this step still matched exactly: the plain cell against Step 1, and all six booklet fixes against the plain cell.
  - The hands-rule runs can't be checked that way.
- **No other departure:** no frozen file was edited, and no rule, threshold or seed changed after approval.
