# Stage H1: hidden-rule worlds. Result

**When and how:**
- **When:** 10:50 to 11:27 GMT on 14 November 2026.
- **Protocol:** `PROTOCOL_H1.md` and its amendments made before any run. The code was committed in `844ccb58` before the first run.
- **Run order:** the four legality checks, then 78 runs at 131,072 steps (21.7 min), then the pre-set fallback for one rule (24 runs at 262,144 steps, 14.2 min), then the analysis.
- **Hardware:** everything trained and was scored on the Mac's GPU, four runs at a time.
- **No AI scientist:** nothing in this step made a model call.

**Sources:**
- Every number below comes from `runs/results.json` (status "final"). The tables are printed from it by `make_report_tables.py`.
- Run times come from `runs/progress.log`.
- Scores are exact-build success on 512 dev designs: H-plain's for the plain world and H1, H2 and H4; H-torus's for H3.

**Integrity:**
- Stage 2b's 39 frozen hashes and all 26 Stage H files were unchanged before the first run and after the analysis.
- No holdout or final file was read.
- Stage 2e stayed paused.

**What ran:**
- 102 runs: 78 main and 24 fallback.
- All finished "ok" on the first attempt. None failed, crashed or timed out.

## Outcome: 2 of 4 rules pass, so Step 2 can go ahead with H1 and H2

| Rule | Budget | Plain, ref A | Plain, fix | Rule, ref A | Rule, fix | Damage, points | Repair, points (95% interval) | p | Holm p | Rule-specific gain, points (95% interval) | Legal | Complete | Passes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1, shifted booklet | 131,072 | 76.6% | 0.0% | 0.0% | 76.6% | 76.6 | +76.6 (+72.5 to +80.4) | 0.0022 | 0.0087 | +153.2 (+147.6 to +158.7) | yes | yes | **yes** |
| H2, mirrored hands | 131,072 | 76.6% | 0.0% | 0.0% | 74.8% | 76.6 | +74.8 (+71.5 to +77.8) | 0.0022 | 0.0087 | +151.4 (+146.2 to +156.2) | yes | yes | **yes** |
| H3, wraparound edges | 131,072 | 76.6% | 74.5% | 51.9% | 64.3% | 24.7 | +12.4 (-1.2 to +24.9) | 0.1320 | 0.2641 | +14.6 (-1.4 to +30.8) | yes | yes | **no** |
| H4, late feedback | 262,144 | 95.5% | 94.3% | 29.4% | 35.6% | 66.1 | +6.2 (-10.3 to +22.4) | 1.0000 | 1.0000 | +7.4 (-9.3 to +23.9) | yes | yes | **no** |

| Rule | Repair at least 25 points with Holm p < 0.05 | Rule-specific gain at least 20 points | Legal | Complete | Recovery fraction |
|---|---|---|---|---|---|
| H1, shifted booklet | yes | yes | yes | yes | 1.00 |
| H2, mirrored hands | yes | yes | yes | yes | 0.98 |
| H3, wraparound edges | no | no | yes | yes | 0.50 |
| H4, late feedback | no | no | yes | yes | 0.09 |

- **The pre-set decision:** with 2 rules passing, Step 2 can go ahead with the shifted booklet (H1) and mirrored hands (H2). Step 3 would need more rules calibrated first.
- **What this doesn't change:** H3 and H4 aren't redesigned or retuned under this protocol. A new rule needs a new protocol.

## Rule by rule

**H1, shifted booklet: passes.**
- Under the rule, reference A built 0.0% of designs on all six seeds.
- With the fix it built 76.6%, exactly reference A's plain-world score.
- It matched design by design on all six seeds, as the construction requires: once fixed, H1's input *is* the plain input.
- **Repair:** +76.6 points (95% interval +72.5 to +80.4), Holm p 0.0087. Two-sided p 2/924 is the smallest a 6-against-6 test can give.
- **On the plain world:** the fix built 0.0%, so the rule-specific gain is +153.2 points.

**H2, mirrored hands: passes.**
- Under the rule, reference A built 0.0% on all six seeds.
- With the fix (flipping the input left to right) it built 74.8%, recovering 98% of the 76.6-point loss.
- **Repair:** +74.8 points (+71.5 to +77.8), Holm p 0.0087.
- **On the plain world:** the fix built 0.0%, so the rule-specific gain is +151.4 points.

**H3, wraparound edges: fails rules 1 and 2.**
- **Damage:** under the rule, reference A built 51.9%, 24.7 points below the plain world. That's just under the 25-point fallback trigger, so no fallback ran.
- **Repair:** the fix (circular padding) built 64.3%, +12.4 points (−1.2 to +24.9), p 0.132, Holm p 0.264.
- **Rule-specific gain:** +14.6 points (−1.4 to +30.8).
- **Seed spread:** the fix's runs varied widely between seeds (SD 16.6 points, 39.8% to 80.3%).
- **The rule leaves a visible trace:**
  - 56.8% of reference A's wrong cells in the wraparound world lie on the edge rows and columns, which are 43.8% of the board. In the plain world the share is 21.5%.
  - With the fix, the share fell to 42.9%.

**H4, late feedback: fails rule 1, on the fallback runs.**
- **At 131,072 steps:**
  - Under the rule, reference A built 0.1% and the fix (γ 0.995, λ 0.99) built 0.6%: a repair of +0.5 points.
  - On the plain world the same recipe built 43.8%, 32.8 points below reference A's 76.6%. The long-horizon recipe learned more slowly at this budget.
  - Damage was 76.5 points and rule 1 failed, so the pre-set fallback ran.
- **At 262,144 steps:**
  - Plain world: reference A 95.5%, fix 94.3%.
  - Rule world: reference A 29.4%, fix 35.6%.
  - Repair +6.2 points (−10.3 to +22.4). The raw p of 0.504 doubles to 1.0 for the second look, and Holm p is 1.0.
- **What it means:** late feedback hurts a lot (66.1 points at 262,144 steps), but the fix chosen in advance doesn't repair it.

## Legality checks

| Rule | Passed | Untrained: exact | Untrained: clean fill | After the short run: exact | Short-run updates | Error |
|---|---|---|---|---|---|---|
| H1, shifted booklet | yes | 0.0% | 2.1% | 0.0% | 7 |  |
| H2, mirrored hands | yes | 0.0% | 1.9% | 0.0% | 7 |  |
| H3, wraparound edges | yes | 0.0% | 1.5% | 0.0% | 7 |  |
| H4, late feedback | yes | 0.0% | 2.1% | 0.0% | 7 |  |

All four fixes are inside the space the AI scientist may search. Each passed the static rules, the parameter and compute caps (each has reference A's 55,846 parameters and FLOP ratio 1.0433), and the learned-policy checks.

## Also recorded (descriptive, no decisions)

Cells (per-seed scores in seed order 4000-4005; exact rate on designs with hidden cells; median training time):

| Cell | Runs ok | Mean | SD | Per seed | Exact on hidden-cell designs | Median training s | Attempts |
|---|---|---|---|---|---|---|---|
| b131072/plain_A | 6/6 | 76.6% | 5.4 | 73.0%, 83.2%, 80.9%, 78.9%, 75.0%, 68.6% | 72.3% | 64.78 | [1, 1, 1, 1, 1, 1] |
| b131072/plain_F_shift | 6/6 | 0.0% | 0.0 | 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0% | 0.0% | 66.92 | [1, 1, 1, 1, 1, 1] |
| b131072/shift_A | 6/6 | 0.0% | 0.0 | 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0% | 0.0% | 62.45 | [1, 1, 1, 1, 1, 1] |
| b131072/shift_F | 6/6 | 76.6% | 5.4 | 73.0%, 83.2%, 80.9%, 78.9%, 75.0%, 68.6% | 72.3% | 67.03 | [1, 1, 1, 1, 1, 1] |
| b131072/plain_F_mirror | 6/6 | 0.0% | 0.0 | 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0% | 0.0% | 63.55 | [1, 1, 1, 1, 1, 1] |
| b131072/mirror_A | 6/6 | 0.0% | 0.0 | 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0% | 0.0% | 62.27 | [1, 1, 1, 1, 1, 1] |
| b131072/mirror_F | 6/6 | 74.8% | 4.3 | 68.0%, 71.3%, 78.7%, 78.7%, 75.2%, 77.0% | 70.2% | 60.59 | [1, 1, 1, 1, 1, 1] |
| b131072/plain_F_torus | 6/6 | 74.5% | 11.9 | 52.9%, 85.2%, 76.6%, 85.4%, 73.4%, 73.2% | 69.8% | 76.44 | [1, 1, 1, 1, 1, 1] |
| b131072/torus_A | 6/6 | 51.9% | 7.8 | 43.6%, 41.0%, 58.6%, 57.0%, 52.5%, 58.6% | 44.2% | 60.25 | [1, 1, 1, 1, 1, 1] |
| b131072/torus_F | 6/6 | 64.3% | 16.6 | 39.8%, 62.1%, 78.5%, 50.0%, 75.2%, 80.3% | 58.7% | 75.47 | [1, 1, 1, 1, 1, 1] |
| b131072/plain_F_late | 6/6 | 43.8% | 19.7 | 24.2%, 73.0%, 63.5%, 34.0%, 37.5%, 30.7% | 33.8% | 60.2 | [1, 1, 1, 1, 1, 1] |
| b131072/late_A | 6/6 | 0.1% | 0.3 | 0.0%, 0.0%, 0.8%, 0.0%, 0.0%, 0.0% | 0.0% | 60.39 | [1, 1, 1, 1, 1, 1] |
| b131072/late_F | 6/6 | 0.6% | 0.9 | 1.4%, 0.0%, 0.0%, 0.2%, 0.0%, 2.0% | 0.0% | 61.24 | [1, 1, 1, 1, 1, 1] |
| b262144/plain_A | 6/6 | 95.5% | 0.7 | 94.3%, 95.3%, 95.9%, 95.5%, 96.3%, 95.9% | 94.7% | 128.23 | [1, 1, 1, 1, 1, 1] |
| b262144/plain_F_late | 6/6 | 94.3% | 1.6 | 91.6%, 93.4%, 95.9%, 94.5%, 95.3%, 95.3% | 93.3% | 145.73 | [1, 1, 1, 1, 1, 1] |
| b262144/late_A | 6/6 | 29.4% | 14.5 | 47.7%, 26.4%, 29.3%, 43.4%, 8.2%, 21.5% | 19.2% | 146.46 | [1, 1, 1, 1, 1, 1] |
| b262144/late_F | 6/6 | 35.6% | 17.6 | 52.9%, 50.6%, 19.3%, 48.2%, 11.7%, 30.9% | 25.4% | 144.06 | [1, 1, 1, 1, 1, 1] |

Where the wrong cells are (share in edge rows and columns 0 and 7; the edges are 43.8% of the board):

| Cell | Wrong cells (all seeds) | Share on the edges |
|---|---|---|
| b131072/plain_A | 3634 | 21.5% |
| b131072/plain_F_shift | 75054 | 32.6% |
| b131072/shift_A | 75054 | 32.6% |
| b131072/shift_F | 3634 | 21.5% |
| b131072/plain_F_mirror | 74938 | 32.6% |
| b131072/mirror_A | 75058 | 32.6% |
| b131072/mirror_F | 3585 | 22.2% |
| b131072/plain_F_torus | 3713 | 23.4% |
| b131072/torus_A | 5823 | 56.8% |
| b131072/torus_F | 4719 | 42.9% |
| b131072/plain_F_late | 8382 | 25.9% |
| b131072/late_A | 60531 | 39.7% |
| b131072/late_F | 44460 | 32.3% |
| b262144/plain_A | 788 | 25.1% |
| b262144/plain_F_late | 1083 | 25.6% |
| b262144/late_A | 16130 | 30.4% |
| b262144/late_F | 14197 | 27.3% |

H1 identity (shift fix against plain reference A, same seeds): all per-design results identical: True
- seed 4000: plain_A 73.0%, shift_F 73.0%, same per design True
- seed 4001: plain_A 83.2%, shift_F 83.2%, same per design True
- seed 4002: plain_A 80.9%, shift_F 80.9%, same per design True
- seed 4003: plain_A 78.9%, shift_F 78.9%, same per design True
- seed 4004: plain_A 75.0%, shift_F 75.0%, same per design True
- seed 4005: plain_A 68.6%, shift_F 68.6%, same per design True

Reference A on plain W_s: 76.6% here (hplain_dev, GPU) against 69.0% in 2c's calibration (ws_dev, CPU); permutation p 0.0931. 2c scored ws_dev on the CPU; plain_A scores hplain_dev on the GPU. Descriptive only.

Learning curves (mean training-episode exact rate at the last logged update):

- b131072/plain_A: 65.3% at 122,880 steps
- b131072/plain_F_shift: 0.0% at 122,880 steps
- b131072/shift_A: 0.0% at 122,880 steps
- b131072/shift_F: 65.3% at 122,880 steps
- b131072/plain_F_mirror: 0.0% at 122,880 steps
- b131072/mirror_A: 0.0% at 122,880 steps
- b131072/mirror_F: 63.6% at 122,880 steps
- b131072/plain_F_torus: 63.2% at 122,880 steps
- b131072/torus_A: 42.0% at 122,880 steps
- b131072/torus_F: 57.0% at 122,880 steps
- b131072/plain_F_late: 34.3% at 122,880 steps
- b131072/late_A: 0.0% at 122,880 steps
- b131072/late_F: 0.8% at 122,880 steps
- b262144/plain_A: 91.3% at 249,856 steps
- b262144/plain_F_late: 89.7% at 249,856 steps
- b262144/late_A: 26.0% at 249,856 steps
- b262144/late_F: 31.2% at 249,856 steps

## What this can and can't show

| Comparison | Shows here | Doesn't show |
|---|---|---|
| Rule world against plain world, both with reference A | All four rules hurt reference A: H1 and H2 to 0%, H4 to 0.1% at 131,072 steps, H3 by 24.7 points | Why H3 and H4 hurt as they do |
| Fix against reference A, on the rule world | For H1 and H2, one known, legal change wins back almost all of the loss | That the AI scientist can find it. That's Step 2. |
| Gain on the rule world against gain on the plain world | For H1 and H2 the gain belongs to the rule; on the plain world the same change gives 0% | For H3, how much of the gap comes from the different design set |
| Legality checks | All four fixes are allowed | That any fix is the best possible change |
| All four together | Two worlds are usable for Step 2 | Anything about self-improvement, since Step 1 has no AI scientist |

## Observations for Step 2 (not decisions)

- **Both passing rules are all-or-nothing:**
  - Under H1 and H2, reference A learns to stop at once. It pressed DONE on every dev design on all six seeds and placed nothing (IoU 0.0), so its wrong cells are exactly the designs' own cells (75,054 over six seeds).
  - The same happens on the plain world with either fix.
  - A search that sees one score per seed would see the collapse, and a richer readout would show the agent giving up, but neither points to the cause.
  - So Step 2's readouts decide whether the cause can be found from evidence or only guessed. (IoU and DONE rates are from each run's `result.json`.)
- **Both fixes are the same kind of change:** they re-index the input before the first layer. Step 2 with H1 and H2 tests one kind of discovery, not two.
- **H3 does leave a clear trace** (errors gather on the wrapped edges), but its fix didn't pass. It could come back as a Step 2 rule only under a new protocol, for example with a different fix or budget.

## Departures and disclosures

- **Amendments:** all are in `PROTOCOL_H1.md`, made before any run.
  - The GPU is used throughout, at the user's request. The frozen evaluator ran on the CPU only as the reference in the identity test.
  - The trainer's identity test used 32,768 steps.
  - The evaluator's identity test used 2c's calibration checkpoint.
  - The short run is 7 updates of 1,024 steps.
- **The rule-specific gain for H1 and H2 is above 100 points** because the fixes take the plain world to 0%. That rule only asks that most of the gain come from the rule, which it plainly does.
- **Reference A's plain-world score here (76.6%) is higher than 2c's 69.0%** (permutation p 0.093). The designs differ (H-plain against W_s's dev set) and so does the scoring device, which the identity test showed doesn't change scores on 256 check designs. Every Stage H cell uses the same designs and evaluator, so the comparisons above aren't affected.
- **Nothing was rerun or retuned** beyond the pre-set fallback.

## Full tables

Every table above, as printed by `make_report_tables.py` from `runs/results.json`:

Status: final; written 2026-11-14 11:26:41 UTC; 102 runs recorded; 36.5 min from the checks to the end of the analysis.
Preflight: {'stage2b_frozen': 39, 'stage2b_changed': [], 'stageH_files': 26, 'stageH_changed': [], 'ok': True}
Postflight: {'stage2b_frozen': 39, 'stage2b_changed': [], 'stageH_files': 26, 'stageH_changed': [], 'ok': True}

Fallback triggered for: ['late']

| Rule | Budget | Plain, ref A | Plain, fix | Rule, ref A | Rule, fix | Damage, points | Repair, points (95% interval) | p | Holm p | Rule-specific gain, points (95% interval) | Legal | Complete | Passes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1, shifted booklet | 131,072 | 76.6% | 0.0% | 0.0% | 76.6% | 76.6 | +76.6 (+72.5 to +80.4) | 0.0022 | 0.0087 | +153.2 (+147.6 to +158.7) | yes | yes | **yes** |
| H2, mirrored hands | 131,072 | 76.6% | 0.0% | 0.0% | 74.8% | 76.6 | +74.8 (+71.5 to +77.8) | 0.0022 | 0.0087 | +151.4 (+146.2 to +156.2) | yes | yes | **yes** |
| H3, wraparound edges | 131,072 | 76.6% | 74.5% | 51.9% | 64.3% | 24.7 | +12.4 (-1.2 to +24.9) | 0.1320 | 0.2641 | +14.6 (-1.4 to +30.8) | yes | yes | **no** |
| H4, late feedback | 262,144 | 95.5% | 94.3% | 29.4% | 35.6% | 66.1 | +6.2 (-10.3 to +22.4) | 1.0000 | 1.0000 | +7.4 (-9.3 to +23.9) | yes | yes | **no** |

Decision: **step2_only**; 2 of 4 pass: ['shift', 'mirror'].

| Rule | Repair at least 25 points with Holm p < 0.05 | Rule-specific gain at least 20 points | Legal | Complete | Recovery fraction |
|---|---|---|---|---|---|
| H1, shifted booklet | yes | yes | yes | yes | 1.00 |
| H2, mirrored hands | yes | yes | yes | yes | 0.98 |
| H3, wraparound edges | no | no | yes | yes | 0.50 |
| H4, late feedback | no | no | yes | yes | 0.09 |

Legality checks (fix in its rule world, check designs, GPU):

| Rule | Passed | Untrained: exact | Untrained: clean fill | After the short run: exact | Short-run updates | Error |
|---|---|---|---|---|---|---|
| H1, shifted booklet | yes | 0.0% | 2.1% | 0.0% | 7 |  |
| H2, mirrored hands | yes | 0.0% | 1.9% | 0.0% | 7 |  |
| H3, wraparound edges | yes | 0.0% | 1.5% | 0.0% | 7 |  |
| H4, late feedback | yes | 0.0% | 2.1% | 0.0% | 7 |  |

Cells (per-seed scores in seed order 4000-4005; exact rate on designs with hidden cells; median training time):

| Cell | Runs ok | Mean | SD | Per seed | Exact on hidden-cell designs | Median training s | Attempts |
|---|---|---|---|---|---|---|---|
| b131072/plain_A | 6/6 | 76.6% | 5.4 | 73.0%, 83.2%, 80.9%, 78.9%, 75.0%, 68.6% | 72.3% | 64.78 | [1, 1, 1, 1, 1, 1] |
| b131072/plain_F_shift | 6/6 | 0.0% | 0.0 | 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0% | 0.0% | 66.92 | [1, 1, 1, 1, 1, 1] |
| b131072/shift_A | 6/6 | 0.0% | 0.0 | 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0% | 0.0% | 62.45 | [1, 1, 1, 1, 1, 1] |
| b131072/shift_F | 6/6 | 76.6% | 5.4 | 73.0%, 83.2%, 80.9%, 78.9%, 75.0%, 68.6% | 72.3% | 67.03 | [1, 1, 1, 1, 1, 1] |
| b131072/plain_F_mirror | 6/6 | 0.0% | 0.0 | 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0% | 0.0% | 63.55 | [1, 1, 1, 1, 1, 1] |
| b131072/mirror_A | 6/6 | 0.0% | 0.0 | 0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0% | 0.0% | 62.27 | [1, 1, 1, 1, 1, 1] |
| b131072/mirror_F | 6/6 | 74.8% | 4.3 | 68.0%, 71.3%, 78.7%, 78.7%, 75.2%, 77.0% | 70.2% | 60.59 | [1, 1, 1, 1, 1, 1] |
| b131072/plain_F_torus | 6/6 | 74.5% | 11.9 | 52.9%, 85.2%, 76.6%, 85.4%, 73.4%, 73.2% | 69.8% | 76.44 | [1, 1, 1, 1, 1, 1] |
| b131072/torus_A | 6/6 | 51.9% | 7.8 | 43.6%, 41.0%, 58.6%, 57.0%, 52.5%, 58.6% | 44.2% | 60.25 | [1, 1, 1, 1, 1, 1] |
| b131072/torus_F | 6/6 | 64.3% | 16.6 | 39.8%, 62.1%, 78.5%, 50.0%, 75.2%, 80.3% | 58.7% | 75.47 | [1, 1, 1, 1, 1, 1] |
| b131072/plain_F_late | 6/6 | 43.8% | 19.7 | 24.2%, 73.0%, 63.5%, 34.0%, 37.5%, 30.7% | 33.8% | 60.2 | [1, 1, 1, 1, 1, 1] |
| b131072/late_A | 6/6 | 0.1% | 0.3 | 0.0%, 0.0%, 0.8%, 0.0%, 0.0%, 0.0% | 0.0% | 60.39 | [1, 1, 1, 1, 1, 1] |
| b131072/late_F | 6/6 | 0.6% | 0.9 | 1.4%, 0.0%, 0.0%, 0.2%, 0.0%, 2.0% | 0.0% | 61.24 | [1, 1, 1, 1, 1, 1] |
| b262144/plain_A | 6/6 | 95.5% | 0.7 | 94.3%, 95.3%, 95.9%, 95.5%, 96.3%, 95.9% | 94.7% | 128.23 | [1, 1, 1, 1, 1, 1] |
| b262144/plain_F_late | 6/6 | 94.3% | 1.6 | 91.6%, 93.4%, 95.9%, 94.5%, 95.3%, 95.3% | 93.3% | 145.73 | [1, 1, 1, 1, 1, 1] |
| b262144/late_A | 6/6 | 29.4% | 14.5 | 47.7%, 26.4%, 29.3%, 43.4%, 8.2%, 21.5% | 19.2% | 146.46 | [1, 1, 1, 1, 1, 1] |
| b262144/late_F | 6/6 | 35.6% | 17.6 | 52.9%, 50.6%, 19.3%, 48.2%, 11.7%, 30.9% | 25.4% | 144.06 | [1, 1, 1, 1, 1, 1] |

Where the wrong cells are (share in edge rows and columns 0 and 7; the edges are 43.8% of the board):

| Cell | Wrong cells (all seeds) | Share on the edges |
|---|---|---|
| b131072/plain_A | 3634 | 21.5% |
| b131072/plain_F_shift | 75054 | 32.6% |
| b131072/shift_A | 75054 | 32.6% |
| b131072/shift_F | 3634 | 21.5% |
| b131072/plain_F_mirror | 74938 | 32.6% |
| b131072/mirror_A | 75058 | 32.6% |
| b131072/mirror_F | 3585 | 22.2% |
| b131072/plain_F_torus | 3713 | 23.4% |
| b131072/torus_A | 5823 | 56.8% |
| b131072/torus_F | 4719 | 42.9% |
| b131072/plain_F_late | 8382 | 25.9% |
| b131072/late_A | 60531 | 39.7% |
| b131072/late_F | 44460 | 32.3% |
| b262144/plain_A | 788 | 25.1% |
| b262144/plain_F_late | 1083 | 25.6% |
| b262144/late_A | 16130 | 30.4% |
| b262144/late_F | 14197 | 27.3% |

H1 identity (shift fix against plain reference A, same seeds): all per-design results identical: True
- seed 4000: plain_A 73.0%, shift_F 73.0%, same per design True
- seed 4001: plain_A 83.2%, shift_F 83.2%, same per design True
- seed 4002: plain_A 80.9%, shift_F 80.9%, same per design True
- seed 4003: plain_A 78.9%, shift_F 78.9%, same per design True
- seed 4004: plain_A 75.0%, shift_F 75.0%, same per design True
- seed 4005: plain_A 68.6%, shift_F 68.6%, same per design True

Reference A on plain W_s: 76.6% here (hplain_dev, GPU) against 69.0% in 2c's calibration (ws_dev, CPU); permutation p 0.0931. 2c scored ws_dev on the CPU; plain_A scores hplain_dev on the GPU. Descriptive only.

Learning curves (mean training-episode exact rate at the last logged update):

- b131072/plain_A: 65.3% at 122,880 steps
- b131072/plain_F_shift: 0.0% at 122,880 steps
- b131072/shift_A: 0.0% at 122,880 steps
- b131072/shift_F: 65.3% at 122,880 steps
- b131072/plain_F_mirror: 0.0% at 122,880 steps
- b131072/mirror_A: 0.0% at 122,880 steps
- b131072/mirror_F: 63.6% at 122,880 steps
- b131072/plain_F_torus: 63.2% at 122,880 steps
- b131072/torus_A: 42.0% at 122,880 steps
- b131072/torus_F: 57.0% at 122,880 steps
- b131072/plain_F_late: 34.3% at 122,880 steps
- b131072/late_A: 0.0% at 122,880 steps
- b131072/late_F: 0.8% at 122,880 steps
- b262144/plain_A: 91.3% at 249,856 steps
- b262144/plain_F_late: 89.7% at 249,856 steps
- b262144/late_A: 26.0% at 249,856 steps
- b262144/late_F: 31.2% at 249,856 steps
