# E2E Q1e: bifurcation diagrams in temperature at fixed alpha (from the fixed start and from e_is), toward T = 0:
# which attractor the 2-cycle region turns into at low temperature (float64 column scans, 160 cells each)
cd /data2/neural-shrimp
L=/data2/neural-shrimp-data/explore/e2e/logs
S="nice -n 10 env CUDA_VISIBLE_DEVICES= .venv/bin/python tools/explore_llm_soft_loop.py scan --threads 4 --dtype float64"
for a in 0.070 0.0628; do
  $S --alpha $a $a --log-temperature -2.0 -0.10 --resolution 160 1 --transient 1000 --samples 300 --record 96 \
     --max-period 48 --tol 1e-8 --projections --out e2e/tcol-a$a > $L/tcol-a$a.log 2>&1
done
