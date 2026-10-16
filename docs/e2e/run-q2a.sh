# E2E Q2a: boundaries of the period-6 band (cuts at fixed alpha, free log10 T) and of the period-3 crescent (cuts at
# fixed log10 T, free alpha); seeds from the fixed start at band/crescent centres read off map cross-512.
cd /data2/neural-shrimp
L=/data2/neural-shrimp-data/explore/e2e/logs
B="nice -n 10 env CUDA_VISIBLE_DEVICES= .venv/bin/python tools/e2e_band.py cut"
for ac in 0.0700:-0.14055 0.0690:-0.14270 0.0710:-0.13935 0.0720:-0.13890 0.0675:-0.14665 0.0670:-0.14850 0.0665:-0.15000; do
  a=${ac%%:*}; c=${ac#*:}
  $B --fixed alpha --value $a --center $c --period 6 --span 0.012 --out band-cut-a$a > $L/band-cut-a$a.log 2>&1
done
for tc in -0.148:0.06500 -0.156:0.06410 -0.142:0.06650 -0.165:0.06400; do
  t=${tc%%:*}; c=${tc#*:}
  $B --fixed log_t --value $t --center $c --period 3 --span 0.004 --step 1e-4 --step-max 2e-4 --out crescent-cut-t$t > $L/crescent-cut-t$t.log 2>&1
done
