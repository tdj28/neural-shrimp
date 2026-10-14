# Task19 postrun saved-result review checklist

Prepared 2026-10-14 from the Root-accepted source; no target arrays, model or
producer analyses were executed to prepare this note. This is a review guide,
not a new auditor, run authorization or changed statistical design.

Source authority: `trajectory.py` SHA256
`8da808f5413d4f7caca49b6c0cba4c8182ff4322c91e123513e9ccb3e9737524`;
`reduce.py` `d87ec765cde3c0cc1145e6225b9b98b9a9c85468df85368c938c571aebda098b`;
`runtime.py` `41f0f19f023f2bbd5f6842929bb07cad70b86ea8d0399eec717a22ae478d468a`.
The prepared trajectory/saved plan hashes were respectively `2c4f76d6…` and
`11d6071d…`; use Inference's actual admitted plan, captured source and original
receipts for the run. Do not substitute prepared authority for actual admission.

## Common completion and provenance

- Read the original supervisor/worker status, exit, timeout, measured wall/RSS,
  output inventories and original process closure. Computational completion is
  distinct from agreement with the scientific claims. Preserve any incomplete,
  failed or negative result; do not select a later attempt without its history.
- Bind the five captured source files, actual reviewed plan, current independent
  ManualQwen model/config/vector pins, and all10 input-authority members. The
  immutable input manifest is `d3014365b8445f8508271959137b922f694f0519665e9dfcb8f459e49517d2e8`.
  Pre/post source/input hashes and trajectory post-asset checks must agree.
- Preserve the snapshot's provenance limit: producer files were writable when
  copied; hash/size/mtime stability across copying and immutable own copies are
  established. Historical producer writer termination is not independently
  attested. Do not invent historical model-byte or execution evidence.
- Check actual backend/thread/CPU/nice/AS/headroom limits against the admitted
  card. Worker checks and RSS/headroom sampling are watchdogs; external owned
  supervision supplies the hard wall deadline.

## Mandatory trajectory001

- Require fixed alpha .0655/log10T .1899492494639028, full896 state, full151936
  vocabulary,24 layers, steering after block12, fixed37-token prefix/window1,
  float64 and the fixed last-prefix-token-embedding start. Exactly1000 one-row
  transient calls plus2000 three-row recorded calls means3000 calls/7000 row
  forwards. Check event ordering/completed counts and every native input/return
  array; future preallocated NaNs must not count as completed data.
- On each recorded row, state is **F(input)** while lead/pmax/entropy are the
  readout of that same **input**. Transient outputs feed the next transient
  inputs; record inputs after the first come from the saved renormalization.
  Full state input/output arrays and leading-probability/entropy/token summaries
  are retained; full151936-probability vectors and logits are not.
- Check both seed0 normalized Gaussian initial directions and absolute offsets
  1e-6/1e-9. Each twin uses the returned difference from the base, raw distance,
  distance floored at epsilon*1e-30, log(distance/epsilon), and corresponding
  renormalization. Read every floor count and all2000 growth rows. Keep the two
  overall means, four disjoint500-step means and20 running means at100-step
  spacing. A floor use or window disagreement must remain visible.
- Read all64 last-state lag distances, historical1e-8*max(1,||last||) detector,
  diameter statistic, observed token period, complete token-cycle IDs and token
  counts. Both twins>.001 and no detected lag-period are finite diagnostics,
  not an asymptotic chaos proof or exclusion of all periodic orbits.
- Verify the **full observed seven-token cycle**, not merely period7 or the
  original distinct-token list. The appropriate producer comparison is the
  admitted `trace.npz` leading-token sequence/cycle, or a correctly pinned
  tokenizer decode. Preserve phase-specific differences; do not infer the
  requested words from q=7. This small exact saved comparison was left outside
  the original worker, so report it separately if performed.
- `primary_seven_phase_clock_matches` and the separate primary qualification
  must agree with q==7. The fixed-seven phase6 fit is retained even if the
  observed period changes, but then does not establish the requested phase6
  relation. Read all fixed-seven phase rows and every actual-q phase row.
- For each phase pair, inspect actual finite-sample nearest state distance and
  returned indices, centroid distance and bounding-ball gap. The scale is mean
  state norm over the recorded trajectory. For q=7 there are21 phase pairs.
  These are different statistics; original plotting source reported centroid
  separation. Separated samples/bounding balls do not prove disjoint invariant
  bands. q=1 has no cross-phase pair and correctly yields undefined separation.
- Read phase6's degree, observation/pair counts, residual/range, unfloored and
  historical-floor exponent, all stationary roots, curvature and derivative
  signs. Read every producer comparison leaf, both twin differences, the
  original band summary and centroid comparison. Keep all nonzero discrepancies,
  including disagreements beyond quoted rounding. Late-state equality is not
  required across independently evaluated chaotic float64 trajectories.

## Saved001: exact all65 cohort and every phase

- Require the immutable design's exact65 ordered (log10T,alpha) pairs,2000x65
  token/probability arrays, token IDs0..151935 and finite float64 probabilities
  in(0,1]. Check original period/lead-period integers0..64, two finite twin
  estimates, offsets1e-6/1e-9 and original map/horizon metadata. Independently
  recomputed token periods must match the original records.
- Read all65 row dispositions, including the original two periodic and one
  irregular case if reproduced. Do not hard-code62 positive cases or suppress
  a changed classification. The saved scalar sequences cannot independently
  replay full-state returns, state-period detection or twin Lyapunov exponents;
  those comparator/classification fields explicitly inherit producer evidence.
- Historical original fit: phase0 observation count determines degree
  min(8,max(2,(observations-1)//6)); each actual phase has its own observation and
  pair counts. Power basis is scaled to[0,1], phase selection minimizes training
  residual/range, and the historical summary's value qualification is<.01.
  Retain original phase choice, coefficients, rank warnings and every comparison.
- Historical amendment: degrees4,6,...24 capped per phase by
  max(4,(pairs-1)//6), NumPy seed0 random fivefold pair split, training-range
  Chebyshev scaling, equal weighting of fold MSEs, minimum CV degree, all-data
  refit, literal CV<.01 phase selection and median of their per-step exponents.
  Read all candidate/fold errors, partitions, ranks, out-of-training-range test
  counts, selected degrees, phase list, exponent spread and summary counts.
  Rank warnings are retained without silently changing the historical selection.
- Historical power-root imaginary tolerance applies to normalized roots;
  Chebyshev's historical tolerance applies to physical-coordinate roots. Read
  `producer_stationary_count` separately from the independent normalized-root
  diagnostics. One stationary root is not automatically one quadratic maximum:
  the latter additionally requires negative curvature and positive-to-negative
  derivative signs. Keep minimum/degenerate/unresolved classifications.
- Independent purged-contiguous sensitivity: all five contiguous heldout pair
  blocks, removal of adjacent training pairs, no shared scalar endpoint across
  train/test, same frozen candidate grid and explicit rank qualification. Read
  every exclusion, undefined derivative and phase estimate. Historical1e-300
  logarithmic-floor comparisons remain separate from unfloored results.
  No full twin exponent may enter degree or phase selection.
- Review every per-cell producer delta and original/amended summary, not just
  49/62 or51/62 totals. Report denominators, missing fits, stationary-count versus
  quadratic-maximum counts and any rank-related selection difference. Retain
  all candidate/fold/phase records even if the purged sensitivity is less favorable.

## Interpretation to preserve

The strongest example was selected and its outcomes were disclosed. The
amendment followed outlier/degree/phase inspection and a rejected spline, even
though the all-cell rule was committed before its amended aggregate. Original,
amended and independent sensitivity results must remain separately labeled.

Low scalar value error does not prove an injective smooth conjugacy, validate
derivatives or require all phase exponents to agree. Random folds and purged
contiguous folds are dependent-sequence interpolation diagnostics, not fresh
confirmation or prospective forecasts; selecting a degree on the reported CV
score adds selection optimism. Finite fitted maxima and recorded token
repetition do not prove exact/global unimodality, Feigenbaum universality,
infinite text repetition or an invariant one-dimensional attractor.

Optional additional trajectories are separate, currently unlaunched: original
outlier index4/log10T .1799135096497498 and minimum-log10T eligible period28
index13/.18478127233738384, alpha .0655. They cannot be substituted for the
mandatory cell or chosen again after own outcomes.
