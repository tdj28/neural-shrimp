# E2E Q1e: bifurcation diagram in temperature at alpha 0.070, every cell started on the 2-cycle state of
# (0.070, -0.19) (init states stored in float32, as scan's warm-start format): which attractor the 2-cycle turns
# into at low temperature. Float64 iteration; 96 cells over log10 T in [-2.0, -0.17].
cd /data2/neural-shrimp
L=/data2/neural-shrimp-data/explore/e2e/logs
nice -n 10 env CUDA_VISIBLE_DEVICES= .venv/bin/python tools/explore_llm_soft_loop.py scan --threads 4 --dtype float64 \
  --alpha 0.070 0.070 --log-temperature -2.0 -0.17 --resolution 96 1 --transient 600 --samples 200 --record 64 \
  --max-period 48 --tol 1e-8 --projections --init-states e2e/init-p2-a0.070 --out e2e/tcol-a0.070-p2start > $L/tcol-a0.070.log 2>&1
