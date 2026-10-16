# E2E Q3a: spectra of Sigma_p and DF along periodic and chaotic orbits (float64, dense forward-mode DF)
cd /data2/neural-shrimp
L=/data2/neural-shrimp-data/explore/e2e/logs
nice -n 10 env CUDA_VISIBLE_DEVICES= .venv/bin/python tools/e2e_rank.py spectra --threads 4 \
  --cell p2-is-it 0.0628 -0.13 2 --cell p3-crescent 0.0641 -0.156 3 --cell p6-band 0.0700 -0.14055 6 \
  --cell p8-stripe 0.071674 -0.115 8 --cell chaos-crescent 0.0652 -0.156 12 --cell chaos-stripes 0.07484 -0.1226 12 \
  --seed p2-is-it=is --transient 1000 --out rank-spectra > $L/rank-spectra.log 2>&1
