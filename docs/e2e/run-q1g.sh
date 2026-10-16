# E2E Q1g: the 2-cycle's lower flip at alpha 0.070, between log10 T -0.4325 (mu -0.677, down record 4) and
# -0.5325 (mu -1.081, down record 5)
cd /data2/neural-shrimp
nice -n 10 env CUDA_VISIBLE_DEVICES= .venv/bin/python tools/e2e_flip.py --branch p2-tcont-a0.070-down --i 4 --j 5 --period 2 \
  --fixed alpha --value 0.070 --out p2-flip-a0.070 > /data2/neural-shrimp-data/explore/e2e/logs/p2-flip-a0.070.log 2>&1
