# E2E Q1d: the 2-cycle continued in T at alpha 0.0628 (moved from run-q1a to a 4-thread lane)
cd /data2/neural-shrimp
L=/data2/neural-shrimp-data/explore/e2e/logs
R="nice -n 10 env CUDA_VISIBLE_DEVICES= .venv/bin/python tools/e2e_branches.py tcont --threads 4"
$R --alpha 0.0628 --log-t -0.13 --period 2 --seed-token is --to -2.0 --step 0.02 --step-max 0.1 --through --out p2-tcont-a0.0628-down > $L/p2-tcont-a0.0628-down.log 2>&1
$R --alpha 0.0628 --log-t -0.13 --period 2 --seed-token is --to -0.05 --step 0.004 --step-max 0.01 --out p2-tcont-a0.0628-up > $L/p2-tcont-a0.0628-up.log 2>&1
