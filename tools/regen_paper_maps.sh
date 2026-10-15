#!/bin/bash
# Regenerate the paper figures drawn by explore_llm_soft_loop.py render and plot_z3_curves.py. The commands were
# recovered on 15 Oct 2026 by reproducing every old figure byte for byte with the renderer as committed before the
# change (numeric colour bars; chaos drawn by the criterion: exponent above 1e-3 and no period).
set -euo pipefail
cd /data2/neural-shrimp
R="uv run --no-sync python tools/explore_llm_soft_loop.py render"
P=llm-soft-loop/pod64
D=../neural-shrimp-data/explore/$P
$R --scan $P/cross-512 --style lyapunov --title "512 x 512 cells: lower ends of the stripes and the crescent" --name paper-fig-cross512-lyapunov.png
$R --scan $P/zoom-Z2-256 --style lyapunov --title "zoom on the period-8 window" --name paper-fig-zoom-z2-lyapunov.png
$R --scan $P/zoom-Z3-256 --style lyapunov --title "zoom on the period-6 hooks" --name paper-fig-zoom-z3-lyapunov.png
$R --scan $P/scale15/zoom-256 --style lyapunov --title "Qwen2.5-1.5B, 256 x 256 cells" --name paper-fig-qwen15-zoom-lyapunov.png
$R --scan $P/map-stripes-256 --style lyapunov --title "Qwen2.5-0.5B, 256 x 256 cells, with two fold curves (white box: Fig. 1)" --curves $P/curve-p2-fold-plus $P/curve-p2-fold-minus llm-soft-loop/curve-p8-up-plus --box 0.0625 0.0775 -0.17 -0.125 --name paper-fig-stripes-lyapunov-curves.png
$R --scan $P/map-stripes-256 --title "periods, with the period-8 fold curve" --curves llm-soft-loop/curve-p8-up-plus --name paper-fig-stripes-period-fold.png
$R --scan $P/cross-512 --families 12 --title "the cells of Fig. 1, colored by the attractor's text" --name paper-fig-cross512-families.png
$R --scan $P/zoom-Z2-256 --families 10 --title "zoom on the period-8 window: attractor text" --name paper-fig-zoom-z2-families.png
$R --scan $P/zoom-Z3-256 --families 10 --title "zoom on the period-6 hooks: attractor text" --name paper-fig-zoom-z3-families.png
$R --scan $P/scale15/zoom-256 --families 10 --title "Qwen2.5-1.5B: attractor text" --name paper-fig-qwen15-zoom-families.png
for pair in cross-512:cross512-lyapunov zoom-Z2-256:zoom-z2-lyapunov zoom-Z3-256:zoom-z3-lyapunov scale15/zoom-256:qwen15-zoom-lyapunov map-stripes-256:stripes-lyapunov-curves map-stripes-256:stripes-period-fold cross-512:cross512-families zoom-Z2-256:zoom-z2-families zoom-Z3-256:zoom-z3-families scale15/zoom-256:qwen15-zoom-families; do
  scan=${pair%%:*}; name=${pair#*:}
  cp $D/$scan/paper-fig-$name.png paper/llm/figures/fig-$name.png
done
uv run --no-sync python tools/plot_z3_curves.py paper/llm/figures/fig-zoom-z3-curves.png $P/scan32/z3bg-b 8
