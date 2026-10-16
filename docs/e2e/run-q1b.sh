# E2E Q1b: the 2-cycle at alpha 0.070 flips going down in T (attempt q1a, |mu| = 1.08 at log10 T = -0.5325);
# continue the (unstable) 2-cycle through that flip toward T = 0 to see whether it regains stability at low T
cd /data2/neural-shrimp
L=/data2/neural-shrimp-data/explore/e2e/logs
nice -n 10 env CUDA_VISIBLE_DEVICES= .venv/bin/python tools/e2e_branches.py tcont --alpha 0.070 --log-t -0.5325 --period 2 \
  --seed-state /data2/neural-shrimp-data/explore/e2e/p2-tcont-a0.070-down/state-last.npy --transient 0 --to -2.0 \
  --step 0.02 --step-max 0.08 --through --out p2-tcont-a0.070-through > $L/p2-tcont-a0.070-through.log 2>&1
