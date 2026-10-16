# E2E Q3d: the crescent. Chaotic section at (0.0652, -0.156) (existence / question at the noun phase), then R1
# (invariant curve) and R2 (two-token) maps along log10 T = -0.156 across the cascade (flips 0.0644766567,
# 0.0650493555, 0.0650877194, 0.0650999993; chaos to 0.065278). The it/there R2 scans of Q3b were stopped:
# their partial rows (log red-r2scan-p2-a0.070-T.log) give a superstable reduced map (mu ~ 1e-4) where the full
# 2-cycle has |mu| > 1, so that pair does not carry the 2-cycle's instability (see Q3c for is / 's).
cd /data2/neural-shrimp
L=/data2/neural-shrimp-data/explore/e2e/logs
R="nice -n 10 env CUDA_VISIBLE_DEVICES= .venv/bin/python tools/e2e_reduced.py"
C="0.0640 -0.156 0.0644 -0.156 0.0646 -0.156 0.0648 -0.156 0.0650 -0.156 0.06505 -0.156 0.06509 -0.156 0.06511 -0.156 0.0652 -0.156 0.06525 -0.156"
$R section --threads 4 --alpha 0.0652 --log-t -0.156 --pair existence question --period 3 --transient 2000 --steps 3000 --out red-sec-crescent > $L/red-sec-crescent.log 2>&1
$R r2 --threads 4 --ref red-sec-crescent --pair existence question --period 3 --cells $C --out red-r2-crescent > $L/red-r2-crescent.log 2>&1
$R r1 --threads 4 --ref red-sec-crescent --pair existence question --period 3 --cells $C --out red-r1-crescent > $L/red-r1-crescent.log 2>&1
