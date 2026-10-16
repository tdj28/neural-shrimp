# E2E Q1c: zero-temperature successor map with the is-there cycle; endpoints refined; cycles on an alpha grid
cd /data2/neural-shrimp
L=/data2/neural-shrimp-data/explore/e2e/logs
G="nice -n 10 env CUDA_VISIBLE_DEVICES= .venv/bin/python tools/e2e_greedy.py"; T="--threads 4"
$G scan $T --alpha 0.0 0.12 --n 241 --chunk 32 --out greedy-scan2 > $L/greedy-scan2.log 2>&1
$G refine $T --scan greedy-scan2 --out greedy-refine > $L/greedy-refine.log 2>&1
$G cycles $T --grid 0.03 0.10 141 --out greedy-cycles-grid > $L/greedy-cycles-grid.log 2>&1
