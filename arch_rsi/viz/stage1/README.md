# Stage 1 visual package

These are visualizations of Stage 1's finished data, made after the analysis. None of them is used for any score, selection or test. Every score is exact-build success on Q's 512 dev designs; the held-out designs are still locked.

## The files

| File | What it shows |
|---|---|
| `search_overview.png` / `.svg` | Best-so-far progress chart of all 30 searches (360 designs), one panel per arm |
| `search_overview.mp4` | The same chart building up candidate by candidate, then the confirmation and the outcome (25.5 s, 1600×900) |
| `search_closeup.png` / `.svg` | One search from each Claude arm up close, with each new best named, like a best-so-far progress chart |
| `stage1_learning.mp4` | Three networks learning to build one design, at every 10% of training (51.8 s, 1920×1080) |
| `stage1_learning_contact_sheet.png` | The same 11 checkpoints as a single still |
| `stage1_learning_ckpt*.png` | Stills from the video |
| `search_data.json`, `learning_data.json` | The numbers behind each figure |
| `search_chart.py`, `render_learning_stage1.py`, `runs/` | The code, and the snapshot training runs |

## How to read the search chart

- **Each dot** is one network design, scored by its mean over 2 training seeds, which is the score each search selects on.
  - **Blue:** it became its search's new best.
  - **Grey:** it didn't.
  - **A cross at 0:** training hit the 600 s limit.
- **The lines:** each thin line is one search's running best, by the frozen incumbent rule (a strictly higher 2-seed mean). The thick line is the mean of the 10 searches' running bests.
- **Where the lines start:** at 35%, the starting network's mean on seeds 0 and 1. That was the bar every search had to beat.
- **The right-hand column** shows each search's pick, retrained on 8 fresh seeds. The black bar is the arm's mean: 75.2% when Claude sees its scores, 73.6% with them hidden, and 60.4% for random picks from the menu.
- **The dashed line** is the starting network on 12 fresh seeds: 22.7%.
- **For the scores-hidden arm,** Claude never saw any of these scores. The harness recorded them, and the final pick still used them.
- **The drop** from each arm's running best (81%, 85% and 74%) to its fresh-seed mean is selection optimism: the best of 12 noisy tries looks better than it is.

## Fixed rules for the close-up and the video

- **Which searches:** in each Claude arm, the search ranked 6th of 10 by its 8-seed confirmed score, the lower of the two middle searches. That is g0 search 2 (pick r4c1, 78.9%) and no-feedback search 3 (pick r3c1, 77.0%).
- **Which seed:** 1001, the first confirmation seed, for all three networks in the video.
- **Which design:** the first dev design with hidden cells, the same rule as the pilot video.
- **The check:** each network's last snapshot must reproduce the dev score of its Stage 1 confirmation run exactly. The renderer stops if it doesn't.

## Caveats to keep with any of these

- They show one task family, at toy scale, on dev designs only.
- The video shows one seed. On its 12 fresh seeds, the starting network learned the hidden-cell skill on 2.
- The search chart's dots are 2-seed scores, which are noisy: the starting network's 12 fresh seeds ranged from 10% to 81%.
- The comparison that counts is the confirmation column. There, seeing scores added +1.5 points, with a 95% interval of −9.1 to +12.0. That is nothing measurable, so Stage 1's gate failed.
