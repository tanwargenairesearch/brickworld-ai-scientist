# Stage 2e: launch approval

Written 10 November 2026, about 21:15 UTC, before the full run started.

## What is approved

The full Stage 2e run, option A of `PROTOCOL_2E.md`: `run_2e.py --preset 2e`, campaign folder `arch_rsi/campaigns/stage2e`.

## The user's words

The user typed these in the session that drafts the article, not in the session that runs the experiment. That session relayed them here word for word.

- **The go-ahead,** 10 November 2026, about 18:50 UTC: "lets complete stage 2e and then complete pending stages to compete this entire pipeine"
- **The ceilings,** 10 November 2026: asked "Set Stage 2e's limits at 50 machine hours and $230 of notional LLM use?", the user answered "Yes, set both".

## The limits the runner reads

ceiling_hours: 50
ceiling_usd: 230
protocol_sha256: 912b3f321e094227b9718a489441b410293c5818591db70fb8b5f9896491ce10

- A new pair starts only while both are under their ceilings. Pairs already started finish.
- The hash is of `PROTOCOL_2E.md` as approved. The runner refuses to start if the protocol changes.

## What this doesn't approve

- Stage 2c, the self-rewrite stage.
- Opening W's held-out split. `FINAL_OPENED_W` isn't written by this run.
- Stage 3, or anything else after Stage 2e.
- Raising either ceiling. That needs the user's words again, added below with the date.
