# E2E Q2c: coexistence census where the period-6 band crosses the stripes (band centres interpolated from the cut
# at alpha 0.070: P3 flip -0.13929, P6 flip -0.14143, and map cross-512). Starts: the fixed start; the band's
# period-6 state (cut a0.0700); e_is (the 2-cycle side); the stripe attractor reached from the fixed start 0.008
# above the band; the attractor 0.008 below the band. 800 transient + 500 measured steps (reduced for time), twins at 1e-7.
cd /data2/neural-shrimp
L=/data2/neural-shrimp-data/explore/e2e/logs
E=/data2/neural-shrimp-data/explore/e2e
nice -n 10 env CUDA_VISIBLE_DEVICES= .venv/bin/python tools/e2e_band.py census --threads 4 \
  --cells 0.0697 -0.1411 0.0700 -0.14036 0.0708 -0.1396 \
  --starts band=$E/band-cut-a0.0700/seed-state.npy is=$E/p2-tcont-a0.070-down/state-first.npy \
  --prestart above=0.008 below=-0.008 --prestart-steps 800 --transient 800 --steps 500 --out census > $L/census.log 2>&1
