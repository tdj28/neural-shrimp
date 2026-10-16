# E2E Q2b (replaces the rest of Q2a, which at ~2.5 min per point would take ~9 h): band cuts at three more alphas
# across the stripes and the crescent's cut at log10 T -0.148 (whose P3 flip the paper locates at 0.0654807769)
cd /data2/neural-shrimp
L=/data2/neural-shrimp-data/explore/e2e/logs
B="nice -n 10 env CUDA_VISIBLE_DEVICES= .venv/bin/python tools/e2e_band.py cut --threads 4"
for ac in 0.0690:-0.14270 0.0715:-0.13910; do
  a=${ac%%:*}; c=${ac#*:}
  $B --fixed alpha --value $a --center $c --period 6 --span 0.012 --out band-cut-a$a > $L/band-cut-a$a.log 2>&1
done
$B --fixed log_t --value -0.148 --center 0.06500 --period 3 --span 0.004 --step 1e-4 --step-max 2e-4 --out crescent-cut-t-0.148 > $L/crescent-cut-t-0.148.log 2>&1
