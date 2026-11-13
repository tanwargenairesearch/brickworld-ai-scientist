#!/bin/zsh
# Profile three networks one at a time, alone on the GPU, then check each against its original retest run.
cd /path/to/brickworld-ai-scientist
C=arch_rsi/campaigns/stage1/confirm
for s in 764c7fb529694085 1c9c0e6b521fc0bb ef6b9b744290bd7c; do
  .venv/bin/python -I arch_rsi/gpu/profile/profile_run.py --candidate $C/$s/candidate.py --seed 1001 --out arch_rsi/gpu/profile/$s > /dev/null
  .venv/bin/python -I arch_rsi/evaluate_arch.py --trial arch_rsi/gpu/profile/$s --split dev --out arch_rsi/gpu/profile/$s/dev.json > /dev/null
  echo "done $s"
done
