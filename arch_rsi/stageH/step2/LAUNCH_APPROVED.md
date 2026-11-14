# Step 2 full run: launch approved

- **When:** 14 November 2026, after the user read the dry run's results.
- **The user's words:** "Fix judge, then run"
- **What was approved:**
  1. Change the diagnosis judge to read every round's analysis (done; disclosed in PROTOCOL_STEP2.md, "Changes after the dry run").
  2. Then run the full Step 2 run: preset `step2`, 12 searches per arm per world, 96 in all.
- **Expected size, from the dry run:** about 2,700 training runs (about 14 h on the Mac's GPU, four at once) and about 400 calls (about $141 notional).
- **Step 2 manifest at approval:** sha256 of manifest.json = `ae2a051f397337072370200c6ba9b7875d0a5959dd39d59cf5e908a53d0981d9`.
- The dry run (`campaigns/step2_dry`) is excluded from the analysis.
