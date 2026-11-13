#!/bin/zsh
cd /path/to/brickworld-ai-scientist
for n in L_base_s0 L_base_s1 L_big_s0 X_base_s0 X_base_s1 X_big_s0 X_base_1m_s0; do
  ( .venv/bin/python -I arch_rsi/train_arch.py --spec arch_rsi/pilot/$n.json --out arch_rsi/pilot/$n > arch_rsi/pilot/$n.log 2>&1 && \
    .venv/bin/python -I arch_rsi/evaluate_arch.py --trial arch_rsi/pilot/$n --split dev --out arch_rsi/pilot/$n.eval >> arch_rsi/pilot/$n.log 2>&1 ) &
done
wait
echo ALL_DONE
