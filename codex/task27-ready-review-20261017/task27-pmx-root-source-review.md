# PMX source contract and audit boundary

Root source review, 17 October 2026. Producer source is private commit
`32c2f84fe59ab2dbc943886ab0b5bd517c55f0da`. This is source preparation,
not acceptance of any target result or permission to execute an unreviewed card.
The human resumed a bounded round by pointing to Claude's communications.
The requested headline numbers in that communication have already been seen.

Read in full: pi_matrix_analyze.py, pi_matrix2_analyze.py,
pi_pm2cs_analyze.py, pi_f64_reason_summary.py, pi_reason.py,
pi_generate_diag.py, pi_llama_diag_analyze.py and pm2cs-design.md.
Compared pi_reason_f64.py with the previous accepted source: its scientific
change adds skip-groups before the existing group loop, leaving the original
group index and its random direction schedule intact. The actual source and
argument association still require original local receipts.

## Corrected cells and pooled variants

- The two-layout flip is extracted-answer inequality; two missing answers
  agree. Correctness and missingness are separate outcomes. Preserve all three.
- The producer loader retains nonnegative indices, uses the smaller/larger
  layouts from args, pairs only common indices and overwrites duplicate keys.
  The independent adapter must report every duplicate, missing and extra key
  and validate the intended exact cohort before relying on a complete result.
- Clusters come from the larger-layout batch. MATH copies are not independent
  observations; the scored first copy defines one problem. Keep GSM cluster
  resampling separate from the single-problem MATH clusters.
- The original seed is 20261016 and the bootstrap has 10,000 draws. The matrix2
  command consumes one RNG stream in cell CLI order: soft, plain, paired flips,
  paired accuracy, then the pooled primary and requested cross-GPU contrasts.
  Arm/paired clusters follow sorted problem indices; pooled clusters follow
  original row insertion order. Exact CLI and row order therefore matter.
- The pooled endpoint weights by problem count, resampling clusters within
  each primary cell. It is not an unweighted mean of cell rates. Frozen success
  is rate > .05 and lower interval endpoint > .02, with strict inequalities.
- The revised-tokenizer cohort can differ from the old tokenizer's cohort
  because grouping depends on prompt length. Preserve intersections and
  non-overlap; never turn a changed-cohort correction into a paired effect.
- Record each requested original, excluded-cell, GSM-only-corrected and
  both-corrected pool under its actual inputs. An equal count does not show
  equal individual outcomes. Do not infer a cause from numerical equality.

## Cold Stop

- The source uses full-vocabulary softmax(z/T), with logits cast to float32,
  and entropy -sum(p*log(clamp_min(p,1e-12))). Low entropy means strictly below
  .01 nats; equality resets the counter. A count >=256 inserts end-of-thinking.
- The loop checks the thinking-budget condition first, then Cold Stop, then
  the arm's ordinary switch rule. The rule applies to both arms. Its exact
  relationship to upstream b97407c6 remains to be checked against that source.
- The design's three cells list 128+64+64 problems, each in two scored layouts:
  that would be 512 soft and 512 plain rows. The requested '6 of 1024 soft rows'
  must be checked against actual args/rows and cannot be adopted from prose.
- The producer analyzer resets its matrix RNG per cell by a subprocess and
  analyzes CS before the restricted baseline. Its subset is by saved index,
  not a fresh model run. Preserve the exact same-problem sets and source order.
- The claim that every early-finishing row was reproduced exactly is not
  computed by the committed ends/statistics functions. State the actual
  selection predicate, denominator and compared fields; report all discordant
  fields. Saved token/top1 equality is not native full-state bitwise equality.
- A simultaneous change in stopping rule and budget is not an isolated effect
  of either one. Intervals are exploratory, and prior outcomes were exposed.

## QwQ extension

Reuse the accepted independent PM2F saved-scalar reducer only under exact
source/schema compatibility. Audit skip-groups, all 96 new indices, two twins
per problem and disjointness from the original cohort. Retain absent/nonfinite
endpoint or fit dispositions. Endpoint growth is relative to nominal eps0;
positive fitted growth in the selected distance window is not endpoint growth.
Twins share problems. Saved growth rates/native distances remain inherited
measurements, with early-switch and realized-perturbation limitations retained.
Every finite publication difference must be recorded, even below a tolerance.

## Tokenizer diagnosis

The current loader first attempts AutoTokenizer, then uses the fast tokenizer
only if a fixed spaced-text round trip fails, and refuses if that also fails.
Legacy mode bypasses that check. Diagnose only the pinned environment and
assets; one probe is not universal tokenizer correctness.

The diagnosis analyzer reports each condition over the selected indices and
compares its intersection with PM2F by exact McNemar. It reports token budget,
missing answers, forced thinking, no switch, median switch, and tokenizer/GPU/
attention metadata separately. HF generate has a combined budget and its own
attention/default sampling path, so it is not solely a precision intervention.

The producer also decodes answer IDs through two tokenizer implementations.
Reproducing saved scalar counts does not reproduce that decoder check. Either
admit a separately bounded tokenizer-only asset/source operation or explicitly
mark the saved decoder result as inherited. Do not load a model as an audit
shortcut. Do not extract answers from a stored last-600-character suffix as
though it were the original complete answer text.

## Evidence and release

Use the exact local raw archives/retrieval/termination records, source revisions
and input hashes. No other owner's pod queries. Keep all publication fields
either independently compared or explicitly inherited/out of scope. Preserve
failed and partial original attempts and disclose missing historical runtime
authority without inventing hashes. One measured manufactured setup may support
the bounded saved-data card; review the exact independent code/card and private
checkpoint before actual execution, then authenticate owned closure and saved
outputs. Do not rerun models or completed PM2F/E5/Task26 work.
