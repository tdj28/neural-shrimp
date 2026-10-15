## 2026-10-15 00:47 UTC: run (exploratory): TC, exact tangent exponents and QR Lyapunov spectra on the headline 0.5B maps (review response)

- **Question:** The human gave the go about 00:40 UTC ("do it, you have full go"). The question comes from the referee report: do the twin exponents that label the paper's 0.5B maps agree with exact tangent exponents and converge with the horizon, and how many of the leading exponents are positive?
- **Setup:** One pod, one GPU (H100 NVL or PCIe, or A100-80GB), job set tangentcheck, commit of this card, Qwen2.5-0.5B-Instruct (pinned as in H2), float64, philosophy vectors at layer 12.
  - (1) tools/hidden_chaos.py cells on the 448 H2 cells (docs/h2-horizon-cells.json: 96 chaotic, 96 periodic and 32 other cells from each of map-stripes-256 and cross-512), exactly as H2: 1000 transient and 2000 recorded steps, twins at 1e-6 and 1e-9, tolerance 1e-8. Added: --tangent (the tangent is drawn after the twins, so the twins and base orbits are those of a run without it) and running exponents after 250, 500, 1000 and 2000 steps.
  - (2) tools/lyapunov_spectrum.py: the four leading exponents by QR iteration of exact tangents, same horizon, at the 8 cells of docs/tc-spectrum-cells.json. Per map these are the H2-chaotic cells at the 10th, 50th and 90th percentile of the H2 min-offset exponent, and the median H2-periodic cell as a control. The rule was fixed before any TC data.
[Private rate/cost line omitted; exact original source commit and omitted-line hash recorded in authority.]
- **Reading, fixed now:**
  - (a) Agreement: report max |tangent - twin| over all cells and over the H2-chaotic cells, per offset.
  - (b) Labels: tangent-chaotic means a tangent exponent > 1e-3 and no period; count the cells whose label differs from the twin label (both twins > 1e-3, no period).
  - (c) Convergence: report the largest change of the running estimates between 1000 and 2000 steps at chaotic cells.
  - (d) Spectrum: count the positive exponents among the top four at each chaotic cell. One means ordinary chaos; two or more means hyperchaos, which would be reported as a finding.
  - Would surprise us: a tangent-twin difference above 0.05 at any H2-chaotic cell, or label changes beyond cells with exponents near zero.
- **Output:** `../neural-shrimp-data/explore/llm-soft-loop/pod64/hidden/{tc-h2-cells-tangent,tc-spectrum}`
