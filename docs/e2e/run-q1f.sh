# E2E Q1f: the crescent's period-3 loop continued in T at alpha 0.0650 (down through instabilities toward T = 0,
# and up), and at alpha 0.0668 where the greedy loop '\n\nThe question' exists at T = 0
cd /data2/neural-shrimp
L=/data2/neural-shrimp-data/explore/e2e/logs
R="nice -n 10 env CUDA_VISIBLE_DEVICES= .venv/bin/python tools/e2e_branches.py tcont --threads 4"
$R --alpha 0.0650 --log-t -0.148 --period 3 --transient 1500 --to -1.5 --step 0.005 --step-max 0.05 --through --fd --out p3-tcont-a0.0650-down > $L/p3-tcont-a0.0650-down.log 2>&1
$R --alpha 0.0650 --log-t -0.148 --period 3 --transient 1500 --to -0.10 --step 0.002 --step-max 0.005 --fd --out p3-tcont-a0.0650-up > $L/p3-tcont-a0.0650-up.log 2>&1
$R --alpha 0.0668 --log-t -0.1405 --period 3 --transient 1500 --to -1.5 --step 0.005 --step-max 0.05 --through --fd --out p3-tcont-a0.0668-down > $L/p3-tcont-a0.0668-down.log 2>&1
