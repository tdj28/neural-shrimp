# E2E Q3c: the 2-cycle's competing pair at low temperature is ' is' / "'s" at the phase after ' it' (branch q1b:
# both at 0.50 below log10 T -1.1); R2 with that pair, frozen at the 2-cycle state of (0.070, -0.4325), scanned
# along log10 T at alpha 0.070 (full map: stable down to a flip near -0.53, then |mu| grows to ~27 at -1.127)
cd /data2/neural-shrimp
L=/data2/neural-shrimp-data/explore/e2e/logs
E=/data2/neural-shrimp-data/explore/e2e
R="nice -n 10 env CUDA_VISIBLE_DEVICES= .venv/bin/python tools/e2e_reduced.py"
$R section --threads 4 --alpha 0.070 --log-t -0.4325 --pair is s --period 2 --seed-state $E/p2-tcont-a0.070-down/state-4.npy --transient 0 --steps 20 --out red-sec-p2-is-s > $L/red-sec-p2-is-s.log 2>&1
$R r2scan --threads 4 --ref red-sec-p2-is-s --pair is s --period 2 --fixed alpha --value 0.070 --range -1.12 -0.16 --steps 33 --out red-r2scan-p2-is-s-a0.070-T > $L/red-r2scan-p2-is-s-a0.070-T.log 2>&1
