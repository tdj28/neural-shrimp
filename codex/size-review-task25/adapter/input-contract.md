# Task25 six-component input contract — prepared 15 October 2026

This is an input interface for the already reviewed pure reducers, not an
admitted run or an additional experiment. Inference alone binds the actual
manifest, source/receipt authority and operational card. No target payload was
decoded while preparing this adapter. All paths below describe metadata roles;
they do not authorize reading scientific outcomes during preparation.

The CLI is `python -I -B run.py --card CARD --out FRESH_DIRECTORY`. A prepared
card refuses execution. The six adjacent scientific files are `run.py`,
`reducers.py`, `prefix.py`, `start.py`, `plateau.py`, and `depth.py`. The five pure
component modules are byte-identical to their accepted releases; their hashes
are also literal constants in the adapter. The input manifest is a single
path/bytes/SHA256 pin in the card. The card's `source_sha256` maps the six
extension-free filenames to their captured hashes. The card must have schema
`task25-six-component-card-v1` and separately approved status
`reviewed_for_execution`. Resource/capture/closure fields belong to Inference's
external supervisor and are reviewed separately; this source does not grant
itself capacity or a retry.

The actual manifest schema is `task25-six-component-inputs-v1`. It contains:

- `pins`: unique `{id,path,bytes,sha256}` rows. A path appears only once, even
  when used by several roles. This includes every selected scientific input,
  selection document, published summary, original source/receipt/closure
  record, and metadata needed for the exact association. Unselected partial
  files/vector files may be authenticated as provenance without being decoded.
- `authority_assertions`: `{file_id,pointer,expected}` literal assertions into
  these pinned metadata files. The expected fields come from actual original
  authority, not a fabricated generic `passed` field. `authority_basis` and
  `limitations` preserve the complete finite provenance and failure scope.
- `roles`: exactly `scale`, `gain`, `ps`, `cp`, `plateau`, and `depth`, each an
  explicit ordered descriptor list. There is no directory search or implicit
  substitution. Every descriptor has `available` (Boolean), `reason`,
  `provenance`, and `file_id`; an unavailable role requires a nonempty reason
  and null file ID. Availability does not assert an original successful exit.
- `gain_specs`: the fifteen ordered `{source_id,label,family,model}` mappings.
  `gain_order_scope` states that the declared display/coverage order is
  independent and that historical plotting CLI order is unavailable. Do not
  deduplicate a model or relabel a source based on its numerical outcomes.
- `p3_source_selections`: the disclosed additional source views described below.
- `published_comparisons` and `published_comparison_scope`, described below.

All pins are authenticated before payload decoding and again after component
serialization. All JSON roles compare `args` against `expected_args` and any
explicit metadata-only `{pointer,expected}` assertions. Comparisons distinguish
Booleans from numbers. All NPZ loads use `allow_pickle=False`; each role lists
the exact `members` as `{member_name:{shape,dtype}}` with `dtype` equal to
NumPy's literal `dtype.str`. Member inventory, shapes, dtypes and `expected_axes`
must match. No raw object arrays are accepted. Native/model assets are never
loaded by this adapter.

## Scale and gain

Scale IDs must equal the union of all primary and supplemental source IDs in
the accepted fifteen-model roster. Current metadata inventories 29 sources.
Each descriptor includes `expected_rows`, `alpha:0`, `max_period`, exact NPZ
members/axes, and `completion`. NPZ fields are `log10_temperature`, `alpha`,
`lyapunov`, `period`, and `entropy`, with any additional original members
explicitly inventoried. A singleton exact-zero alpha and exactly one scalar
L/P/H per temperature are required. The literal source's flattening is retained;
no multi-alpha projection is silently selected.

`completion` is either `{kind:"all_saved_rows"}` under original final-coverage
authority, or `{kind:"recorded_prefix",count:N}` under an exact saved prefix.
Any saved `done` must agree with the prefix rule. All allocated scalar rows are
still passed to the accepted reducer and retain their explicit completion
flags; unfinished rows cannot silently become qualified. The original literal
summary and finite qualification remain separate. Missing optional versus
required sources follow the unchanged pure reducer, not an invented fallback.

Gain descriptors use their separately assigned source IDs, complete original
JSON args, `expected_rows`, `completion`, and ordered `expected_coordinates`
(at least `log10_temperature`). The native `cells` list is normalized only by
selecting the source fields `log10_temperature`, `entropy_mean`, `log_g_net`,
`log_g_read`, `lyapunov`, `period`, and `boundary_term`, then attaching the
explicit completion flag. All fifteen source files currently declare 24 rows.
Every bin, row, duplicate label, missing/ambiguous model, and P3 comparison
remains in the unchanged reducer output. No power statistics are reconstructed.

The separately disclosed P3 selector amendment retains that full fifteen-source
result, including its ambiguous default Qwen7B reference. It adds
`gain.source_selected_P3_views` from exactly two ordered manifest objects:
`{name,selector_basis,exposure,entries}` named `manuscript_wide7B` and
`historical_narrow_qwen7B`. Each `entries` list has six exact ordered roles:

| Role | Family/model | Wide manuscript source | Historical narrow view |
|---|---|---|---|
| qwen_reference | Qwen2.5/7B | qwen7b-wide | qwen7b |
| qwen14 | Qwen2.5/14B | qwen14b-wide | same |
| qwen32 | Qwen2.5/32B | qwen32b-wide | same |
| olmo_reference | OLMo-2/7B | olmo2-7b-wide | same |
| olmo13 | OLMo-2/13B | olmo2-13b-wide | same |
| olmo32 | OLMo-2/32B | olmo2-32b-wide | same |

Entries are `{role,source_id,family,model}`; source IDs, original global specs,
true family/model identities and availability must agree uniquely. Missing,
duplicate, cross-family or substituted selectors refuse as malformed input;
there is no fallback. Valid selected sources with empty bins still produce
undefined scientific comparisons. Neither labels nor thresholds nor scalar
medians change. Selected views use unchanged `gain_summary` with an explicit
subset and retain their full bin/row/comparison records. Their view-local row
ordinals differ from the parent all-source view and are labelled by source ID.

`selector_basis` is a structured object with authenticated `file_id` and
`manifest_file_id`, exact `source_commit`, positive `source_lines` and
`excerpt_lines`, and `scope`. `exposure` explicitly discloses the prior
manuscript/log/producer-summary exposure and that this is a source-selection
amendment, not fresh blinded prespecification. The frozen current manuscript
expressly invokes the wider Qwen7B run for the6–10nat bin. The historical narrow
view is descriptive and retains its missing bins; it never replaces either the
all-source default or the manuscript-wide view.

## Prefix transitions and loop-free start-state scoring

The forty PS IDs are literally `qwen7b/{0..9}/{coarse,fine}` and
`qwen14b/{0..9}/{coarse,fine}`. Descriptors pin exact original args and ordered
coordinate columns, with 48 coarse and 24 fine cells. They pass the raw cell
records to the accepted reducer. The adaptive first-argmax fine-grid association
is checked there and retained even if it fails; it is not an input-selection
criterion. Development0/1 and held-out2–9 stay fixed. Missing stages and
scientifically unqualified records remain explicit.

The fourteen CP IDs are the accepted `start.FROZEN` model keys. Each descriptor
pins original args and exact `log10T`/`gain_log10T` axes; original prefixes must
be exactly0–9 in order. The reducer retains every saved scalar and profile.
The frozen three-decimal outcome table is the primary score. The adapter passes
the own scale projection as a separately labelled sensitivity analysis, joined
by exact `(family,model)` only. A scale source refusal is passed as such, never
replaced with a rounded paper value. Nonfinite encodings in an own scale
projection are decoded only for this own-to-own scalar handoff; raw evidence is
not altered. Five-decimal profile compatibility stays conditional on the
producer's rounding rule. No native hidden states, logits or random directions
are inferred from profiles.

## Plateau/precision histories and depth scans

The ten plateau IDs and cell counts are PTa24, PTb6, PTc16, B7a_history64,
B7b_history64, QFa29, and QFb_p2/p3/p4/p5 each12. Each descriptor pins args,
exact ordered coordinates and expected count. PTa/PTb additionally require
`selection_file_id`, whose source-defined `meta` entries bind the prior selected
coordinates and map labels. These are inherited selection annotations, not
fresh plateau measurements. The output retains all precision/offset histories,
the fixed FD ladder, strict and inclusive threshold diagnostics, undefined
ratios, original labels, and all failed or unqualified cases. The only extra
cross-run scalar comparison is PTc versus B7b at exact stored coordinate and
shared1e-5 offset; unmatched/duplicate coordinates remain visible.

Depth descriptors have layers20,35,50,65 exactly once each. The first three new
layers20/35/65 require their authenticated final24×24 NPZ plus `summary_file_id`
for the matching scan JSON, with exact original args and axes. Layer50 requires
separate original32×32 authority. If it is still unavailable, it must be an
explicit unavailable descriptor with null file ID and reason; there is no
substitute from the printed4.3% or from the0.5B layer-sweep source. `expected_args`
may be null only for that unavailable descriptor. Final-grid `done` checkpoints
are rejected by the pure depth reducer rather than substituted for final data.
Original tokenizer failures and post-save `period_colors` import failures stay
in each provenance record. The writer's positive-exponent count is separate
from the own period-zero finite qualification. Per-layer vector differences and
24×24 versus32×32 grids remain explicit; no regridding or disappearance claim.

## Saved-summary comparisons and outputs

Each published comparison has `component`, `available`, `file_id`,
`producer_pointer`, `own_pointer`, `source_association`, and `rounding_scope`.
An unavailable comparison carries its reason. Pointer mappings must be frozen
from the exact original schema/source before reading outcomes. Prefer stable
whole projection or score nodes, including null/source-refusal results; do not
pick only favorable leaves. A missing own nested scientific node is recorded
as `own_scientific_node_unavailable`, with every producer leaf marked missing,
while all six full component outputs are retained. A malformed/missing original
pin, original schema or authority still refuses. No rounding transform,
agreement tolerance or unavailable printed full precision is invented. Literal
printed summaries without an authenticated arithmetic/source association are
unavailable or descriptive, not silently reconstructed.

Successful computational completion writes exactly nine files: `scale.json`,
`gain.json`, `ps.json`, `cp.json`, `plateau.json`, `depth.json`,
`comparisons.json`, `input-manifest.json`, and `report.json`. The supervisor may
later append its own final manifest. The report schema is
`task25-independent-six-component-v1` with `computational_status:"complete"`;
scientific failures, missing evidence, undefined scores and discrepancies do
not change that computational disposition. A failure inside a newly created
output writes `failure.json` and raises; an existing directory is never touched.
All nonfinite values have named JSON dispositions rather than finite fill.

Inference retains actual manifest/summary integration. No actual input pins,
execution card, launch, source receipt or target result are asserted here.
