# Stage 2b visual package (the article's video and Figure 7)

Written 9 November 2026, about 22:00 GMT, while Stage 2b was running and before any result was read. The user approved the 4-scene storyboard in `linkedin/single-article-plan.md`. Nothing here is used for any score, selection or test. W's held-out designs stay locked.

**Rendering starts only after `STAGE2B_DONE`.** GPU or CPU load during the run could push its candidates toward the 900 s time limit.

## Files

| File | Scene | Status |
|---|---|---|
| `search_chart_2b.py` | 2 (the search), 3 (the answer), 4 (the limits); also Figure 7 (`search_2b.png`/`.svg`) | Written. Tested on the dry-run campaign as a still only |
| `train_snapshots_2b.py` | 1 (learning) | Written: a line-for-line copy of `stage2/train_2b.py` that also saves 11 weight snapshots. Not run yet |
| scene 1 renderer | 1 | To adapt from `../stage1/render_learning_stage1.py` for W (10×10×6) after 2b finishes, when it can be tested |
| final assembly | 1–4 | ffmpeg concat of the scene files at 1920×1080, 12 fps |

## Fixed rules for scene 1, written before the result

**Networks:**
- The starting network (`candidates/baseline.py`, Stage G's recipe), as in the `start_g` comparator.
- One g0 pick ("Claude sees its scores"), chosen as follows:
  1. Take the g0 searches whose pick isn't the start and was judged compliant by the rule review.
  2. Sort them by 8-seed primary score, highest first.
  3. Take the one at 0-based position ⌊m/2⌋, where m is the number of eligible searches. For 16 searches that's the 9th: the lower of the two middle searches, the same rule as Stage 1's video.
  4. If no g0 search is eligible, scene 1 shows the start alone, and the caption says why.

**Training:**
- Seed 1001, the first confirmation seed.
- World W at Stage 2b's budget of 131,072 steps, compute-matched by `w2b.budget`.
- Device mps, as in Stage 2b.

**Design shown:** the first design in `stage2/targets/w_dev.pt` (dev split) that has hidden cells.

**Scoring:** each snapshot builds with the frozen evaluator's rule (greedy, all 512 dev designs at once). The score printed is exact success over those 512.

**Checks (the renderer stops on any mismatch):**
- the pick's final snapshot scores exactly what its Stage 2b confirmation run scored (`campaigns/stage2b/confirm/<sha>/seed1001/dev.json`);
- the start's final snapshot matches its own confirmation run (`confirm/764c7fb529694085/seed1001/dev.json`).

## Scene 2–4 rules (in `search_chart_2b.py`)

- **Primary scores only:** a pick judged to break a rule scores as the start, and is drawn as a hollow diamond.
- **Every number** is cross-checked against `results.json` (after `review_2b.py`, then `analyze_2b.py`).
- **The script refuses to run** if results aren't final or if `FINAL_OPENED_W` exists.
- **The outcome lines** recompute P1 and P2 from unrounded primary scores. They say "The gate passed" or "The gate failed again" from `primary_family.gate_passed`, with the same wording as `linkedin/rsi-article-v14-2b-endings.md`.

## Caveats to keep with any of these

- One toy world, one model, dev designs only.
- Claude's research program and weights never changed.

## Update, 10 November 2026, before any 2b result was read

The user asked for one frame that shows everything: the brick build, its evaluation, and an autoresearch-style progress chart. There's also a three-loop system diagram (`../stages/system/make_system.py`).

**New scene 1** is `search_build.py`, a prototype tested on Stage 1's g0 search 2.

**Fixed rules for the 2b version:**
- **The search:** the g0 search chosen by the scene-1 rule above (eligible searches, sorted by 8-seed primary score, the one at position ⌊m/2⌋). The same search is shown whatever it looks like.
- **The design:** the first W dev design with hidden cells.
- **The build:** each experiment's seed-0 checkpoint builds it greedily. It must agree with that run's own `per_design` result.
- **The grids:** both seeds' `per_design` results, as stored.
- **Kept:** the frozen incumbent rule, recomputed. It must reproduce `state.json`'s incumbent.
- **Wording:** generic, no model or person names.

**Still to do after STAGE2B_DONE:** wire in W's spec and targets. The script currently uses Stage 1's 8×8×5 world.

## Rendered, 10 November 2026, after `STAGE2B_DONE` and the final analysis

| Output | What it is |
|---|---|
| `search_2b.png` / `.svg` | The article's Stage 2b figure (Figure 6 in the v15 draft; this file's "Figure 7") |
| `search_2b.mp4` | All searches, the answer and the limits (39 s) |
| `build_g0_s15/search_build.mp4` | One search in one frame (35 s) |
| `scene1_system.png` | The three-loop diagram, all three loops shown equally |
| `problem.mp4`, `problem_still.png` | The problem: a target design, and the start and the shown search's pick building it (12 s) |
| `stage2b.mp4` | The article video: the four parts joined by `make_2b_video.py` (96 s, 1920×1080, 12 fps) |

- **The search shown** is g0 search 15, the one the rule above selects: all 16 g0 searches were eligible, and it's 9th by 8-seed primary score (60.1%). `make_2b_video.py` recomputes that choice and stops if the rendered search differs.
- **That search was touched by the interruption:** its round 2 was set aside whole and redrawn before the resume (`../../stage2/2b/REPORT.md`). The rule shows it whatever it looks like.
- **The design shown** is never built exactly in that search; the frames say so.
- **Not rendered:** the original scene 1 (`train_snapshots_2b.py`, a learning clip). The single-frame search view replaced it, as the 10 November update above says.

**Changes made while rendering** (drawing code only; nothing frozen):
- `search_build.py`: a candidate that ties the bar and passes the frozen incumbent rule only by rounding isn't drawn as a new best. The incumbent check against `state.json` still uses the frozen rule.
- `search_build.py`: labels have a surface-coloured outline, and a label that would run off the right edge is drawn level, left of its dot. Subtitles say "a network and its training recipe" when the search wrote recipes.
- `../stages/system/make_system.py`: the default status chips now say Loop 2 is done and Loop 3 wasn't run because the gate failed. The gutter label moves aside when a highlight outline would cross it.

## Storyboard, changed on 10 November 2026 after the result, at the user's request

The user asked for the video to cover the whole process: what the system looks like, what problem it takes on, and how it solves it by searching, with the builds and the improvements on screen. So the video now runs in this order:

| Part | About | What it shows |
|---|---|---|
| 1. The system | 10 s | The three loops, shown equally. The status chips say which ran |
| 2. The problem | 12 s | A target design. The starting network and a network the AI scientist wrote each build it brick by brick |
| 3. How it searches | 35 s | One search in one frame: each experiment's build, its evaluation grid and the progress chart |
| 4. Did the experiments guide it? | 39 s | Every search in every arm, the answer with the gate, and the limits |

**Part 2 was added after the result, so its rule is stated on screen and here:**
- **Networks:** the start (`campaigns/stage2b/baseline/w/seed0`) and the pick of the search shown in part 3, both seed-0 runs.
- **Design:** the first dev design with hidden cells that the pick's run builds exactly and the start's doesn't. That's chosen to show one success beside one failure, so the frame calls it an illustration, not evidence.
- **Rates on screen:** each run's own score over all 512 dev designs, and the confirmed fresh-seed scores from `results.json`.
- **Checks:** each replay must agree with its run's per-design result, and the pick's file hash must match `results.json`.

**Also changed:** the opening diagram no longer highlights Loop 2. Parts 3 and 4 keep the rules written before the result.

## Mobile version, 10 November 2026 (the user asked for one that's easier to watch on a phone)

`make_2b_mobile.py` writes `mobile/stage2b_mobile.mp4`, one still per part (`mobile_1_system.png` to `mobile_5_limits.png`) and `stage2b_mobile_data.json`.

- **The file:** 1080×1350 (4:5), 95.8 s, H.264 High, 4:2:0, 24 frames a second, BT.709 tagged, about 1.2 Mbps (13.9 MB), no audio track, index at the front.
- **Same story, data and checks** as the landscape video: it calls the same loaders (`problem_scene.load`, `search_build.load_search`, `search_chart_2b.load`), the same rule-chosen search and the same designs.
- **Re-laid out, not cropped.** The shared drawing files are imported, not edited: anything the phone needs differently is overridden inside `make_2b_mobile.py`.
- **A cover for the thumbnail:** `mobile/mobile_0_cover.png` (1080×1350), added on 10 November at the user's request ("Make the cover"). It's the 16:9 article cover (`make_cover.py`) re-laid out at 4:5: the same headline, the problem scene's two builds, and 14.2%, 57.8% and 56.3%. It states the builds' selection rule and calls them an illustration. It isn't a frame of the video: upload it as the video's thumbnail, which LinkedIn's help page offers from desktop only.

### Reworked on 10 November 2026, about 18:45 UTC

The user asked for a high-quality phone version, and said: "we must not say gate failed in entire video in any scene or frame, its not correct narrative".

**Wording**
- **No frame says "gate failed".** The Loop 3 chip is "STAGE 2c · NOT RUN". Its card says it runs only if seeing its scores helps Loop 2's search, and that Stage 2b didn't show that.
- **The answer** reads: "The interval includes zero. This test doesn't show that seeing its scores helped the search."
- **The last card** reads: "The next stage, where it rewrites its own research program, hasn't run: this test didn't show that seeing its scores helps."
- **Loop 2's card** says the AI scientist is a hosted LLM. The old sub-line "on one laptop GPU" is gone, because inference is remote.
- **The first frame** carries the cover's headline ("An AI scientist writes the network. Did its experiments guide the search?"), so the thumbnail and the first 3 seconds say what the video is about.

**Legibility** (judged at 390 points wide, a phone's feed width)
- **Smallest text is 28 px**, about 10 points on a phone. Before, 23 labels were 18 to 27 px: chips, axis numbers, the legend, the footers and the "illustration" notice.
- **Part 1:** about 155 words instead of about 195. The footer is folded into Loop 2's card. Each loop is pointed out in turn for 3.5 s, after 1.5 s with all three drawn the same.
- **Part 2:** the "illustration, not evidence" notice and its rule are on screen for the whole scene, not only the last 7 s. Each build's own rates sit under it.
- **Part 3:** the header names the experiment; the chart labels rounds only, in larger type.
- **Part 4:** the two compared arms' searches fill the frame while they run, with a legend. Then the answer has its own frame: 57.8% and 56.3% in large type, the difference and interval, and every arm's picks below. The last card repeats the result above the limits, and adds that the interval still allows a gain of up to +12.2 points.
- **Safe area:** 56 px at the sides and top, 100 px at the bottom.

**Motion and timing**
- **Drawn at 24 frames a second,** not 12 shown twice. Each brick drops in (three frames in part 2, one in part 3), and the evaluation grid fills in 16 steps.
- **Lengths:** system 12.0 s (was 10.0), problem 11.7, one search 35.2, part 4 37.0 (was 39.0: candidate steps 0.75 s each, the last card 11 s). The total is still 95.8 s, so "96 seconds" in the post holds.

**Checks the script runs on itself** (it stops if one fails)
- no text under 28 px, in the drawn frames or the charts;
- nothing drawn outside the safe area, on every frame;
- no on-screen string contains "gate", "LEGO" or a model or vendor name (359 strings checked);
- the answer's numbers are recomputed from unrounded primary scores and must appear in `search_chart_2b.outcome_lines`' own sentence;
- the encoded file is H.264, 4:2:0, 24 frames a second, BT.709, between 384 kbps and 30 Mbps, and 95.5 to 96.5 s long.

### LinkedIn's requirements, read on 10 November 2026

From LinkedIn Help, "Share videos on LinkedIn" (`linkedin.com/help/linkedin/answer/a7174587`) and "Video sharing troubleshooting" (`answer/a548372`). Both pages are marked as two years old.

| Requirement (official, member posts) | Listed value | This file |
|---|---|---|
| Container | MP4 | MP4 |
| File size | 75 KB to 5 GB | 13.9 MB |
| Length | 3 s to 15 minutes | 95.8 s |
| Resolution | 256×144 to 4096×2304 | 1080×1350 |
| Aspect ratio | 1:2.4 to 2.4:1 | 4:5 |
| Frame rate | 10 to 60 frames a second | 24 |
| Bit rate | 192 Kbps to 30 Mbps | about 1.2 Mbps (the earlier render was 180 kbps, under the minimum) |
| Edges | keep key elements, text and logos away from them; no sizes given | 56 px sides and top, 100 px bottom |
| Custom thumbnail | desktop upload flow only | first frame is built to work as one |

**Official, but for ads, not member posts:** 4:5 is the recommended vertical ratio, and 1080×1350 is the largest size listed for it (`linkedin.com/help/lms/answer/a424737`).

**Not stated on any official page I could find:**
- whether the 192 Kbps minimum is enforced;
- whether an audio track is required (this file has none);
- the H.264 profile, pixel format or index position;
- how the default thumbnail is chosen ("the first frame" comes from third-party guides only);
- how a 4:5 video is cropped in the feed or in the full-screen player.

LinkedIn's only safe-zone drawing is for 9:16 full-screen video.

### Still open

- **The landscape video and the shared files still say "gate failed":** `stage2b.mp4`, `search_2b.png`/`.svg`, `search_chart_2b.outcome_lines` and `limits_fig`, and `make_system.py`'s default Loop 3 chip. They weren't edited here because another session was using them.
- **A shorter cut** (for example 45 to 60 s) would suit the feed better than 96 s. It changes the structure, so it needs the user's decision.
