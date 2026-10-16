# E2E Q3b: reduced scalar maps. (1) the 2-cycle: section at its own orbit, R2 (two-token it/there model) scanned
# along log10 T at alpha 0.070 (full map: fold -0.1741065, flip near -0.53) and along alpha at log10 T -0.115
# (full fold alpha_c = 0.06279827); (2) the crescent: chaotic section at (0.0652, -0.156), R1 and R2 maps.
cd /data2/neural-shrimp
L=/data2/neural-shrimp-data/explore/e2e/logs
E=/data2/neural-shrimp-data/explore/e2e
R="nice -n 10 env CUDA_VISIBLE_DEVICES= .venv/bin/python tools/e2e_reduced.py"
$R section --threads 4 --alpha 0.070 --log-t -0.19 --pair it there --period 2 --seed-state $E/p2-tcont-a0.070-down/state-first.npy --transient 0 --steps 20 --out red-sec-p2-a0.070 > $L/red-sec-p2-a0.070.log 2>&1
$R r2scan --threads 4 --ref red-sec-p2-a0.070 --pair it there --period 2 --fixed alpha --value 0.070 --range -0.90 -0.14 --steps 39 --out red-r2scan-p2-a0.070-T > $L/red-r2scan-p2-a0.070-T.log 2>&1
$R r2scan --threads 4 --ref red-sec-p2-a0.070 --pair it there --period 2 --fixed log_t --value -0.115 --range 0.058 0.068 --steps 21 --out red-r2scan-p2-t-0.115-A > $L/red-r2scan-p2-t-0.115-A.log 2>&1
$R section --threads 4 --alpha 0.0652 --log-t -0.156 --pair existence question --period 3 --transient 2000 --steps 3000 --out red-sec-crescent > $L/red-sec-crescent.log 2>&1
