# E2E Q3e: (1) what varies along the crescent's attractor curve; (2) four leading Lyapunov exponents (exact
# tangents, QR) at three chaotic cells, for the Kaplan-Yorke dimension (how one-dimensional the chaos is)
cd /data2/neural-shrimp
L=/data2/neural-shrimp-data/explore/e2e/logs
nice -n 10 env CUDA_VISIBLE_DEVICES= .venv/bin/python tools/e2e_reduced.py along --threads 4 --ref red-sec-crescent --pair existence question --period 3 --out red-along-crescent > $L/red-along-crescent.log 2>&1
nice -n 10 env CUDA_VISIBLE_DEVICES= .venv/bin/python tools/lyapunov_spectrum.py --vectors llm-phase-map/vectors-l12 --concept philosophy \
  --cells-file docs/e2e/lyap-cells.json --cells-index 0 1 2 --k 4 --transient 1000 --record 600 --threads 4 --out e2e/lyap-spectrum > $L/lyap-spectrum.log 2>&1
