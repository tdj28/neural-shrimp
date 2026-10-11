# Independent saved parity-order replay guide

Source-only reviewed producer; independent saved arithmetic remains pending.
No new native model/gradient/HVP/CE calls needed. Full15776-dimensional arrays.

- Raw: `/data2/neural-shrimp-data/explore/tiny-lm-parity-order/attempt001-target`
- Execution: `e8030d33c47bb817b4ac9829951bbb6137315f57`
- Outer: `8e7b47c82b6c25db6873296c83e2a6589a1189e5dc8f3c3a8da2b7af2e69a012`
- Worker: `838bd1102ca3bcd005a10ace5c45bbb2e64902d12080203d1a1e7d5861dbb88f`
- Complete1052outer/1046worker inventory and archived historical sources verified;
  owned2753340/2753432 absent. Runtime43.715630s,971284KiB,224HVP/0gradient/0CE.

## Literal arrays and native event chronology

`run/aliases/<label>/<path>.json` has rows17 exact lossless NPY pointers
`{path,sha256,bytes,dtype,shape}` into `run/objects`. Labels are seed28161original,
seed28161quarter_step, seed28162original, seed28162quarter_step, spelled
`seed-28161-original` etc. Path keys P,Q,T and control-only R.
Use all17 rows, including literal u at0. New original R is not copied: it is the
already independently audited `reverse-trace.npy` in original ordering root
(see below). Derived parity_even/chronological_odd/grouped_odd are one-vector
`array` aliases; recompute them independently instead of accepting producer scalars.

`events/00001-input.json` plus matching `-return.json` through00224 contain
sequence=index,operation=hvp,case=<label>/<path>,exact state,direction and returned
pointers. Require input fields identical in return and all exact immutable bytes.
Sequential path ranges:28161original P1–16,Q17–32,T33–48;28161control P49–64,
Q65–80,T81–96,R97–112;28162original P113–128,Q129–144,T145–160;28162control
P161–176,Q177–192,T193–208,R209–224. Each update is literally v−eta*HVP, with no
renormalization. Exact states must match each time-indexed historical reference.
Orders applied next to u: P0..15; R15..0; Qeven0..14 then odd1..15; T reverseQ.
`<label>/<path>-order.json` stores these indices; independently regenerate them.

`<label>/binding.json` retains exact eta/checkpoint/historical qualifiers. Original
reference/ordered/frozen/readout are from prior response; control arrays from
temporal mean aliases. Original stored response arrays have h1..16 and must
prepend literal gradient direction u (see frozen source N.load_case). Required
readouts are CE gradients at h1/4/16, including ridge removal .001*theta. Existing
independent mean audit validates those inherited comparisons; bind its exact
source/outer/worker in `run/input-binding.json`, do not repeat native operations.

Original R authority: `/data2/neural-shrimp-data/explore/tiny-lm-ordering/attempt001-target`,
execution358a93ca3cc6434d5b8ea5c4989ad391fb471a0f,
outereb3f0c4503a73ac21443fe56b88a3834ec4cf13ae98fc98d53d3e40b4e8fc686,
workerce69af6c687cc66f6cc5118af7744734a8266cc233a6862e34e3d05ea60870ec.
`run/seed-<seed>-arm-256/direction-0-horizon-16/reverse-trace.npy` and state-order
indices15..0; operation_first..last identify all16 old native HVP state/vector/return
files under `run/operations/hvp-NNN-*`. Require bytewise exact states, u, full
update trace and eta. Old independent reversal audit is pinned in ordering-binding.

## Independent arithmetic and interpretation gates

Recompute norms/dots/readouts with math.fsum. E=(Q+T−P−R)/2, oddP=(P−R)/2,
oddQ=(Q−T)/2. Check Q−P=E+(oddQ−oddP), retain cross-term2E·(oddQ−oddP).
Transpose scalar u·(P−R) and u·(Q−T), normalized by ||u||max(||a||,||b||,floor),
threshold1e−8. Vector/squared decomposition threshold1e−12; freshP relative
h1/4/16 replay threshold1e−10, floor1e−12. All FOUR inherited1% plus new replay/
algebra gates required BEFORE primary descriptive: E/||P||≥.20 in BOTHoriginals.
Controls reported under same fixed gates, no post-hoc control-separation gate.
Report all P/R/Q/T gains/CEslopes, E norm/readout, frozen-error fractions, alignment,
and cross terms; no additive attribution percentages. ||u||=1.

This is a linear-response ordering intervention on the same saved16 Jacobians,
not an actual optimizer, data-order or training intervention. It demonstrates
directional sensitivity not present for a commuting sequence, without identifying
individual commutators/basis rotation, explaining the full frozen/mean error,
or proving operator-norm, forecast, training/geometry utility. Four consumed
development cases, no fresh holdout. Original240s profile failure stays preserved;
explicitly authorized300s fresh profile passed without dropping old cold maxima.

## Producer values for comparison only

28161 original:
```json
{
  "P": {
    "norm": 0.5903215289975097,
    "training_ce_slope": -0.10685572631029908
  },
  "Q": {
    "norm": 1440.056352877857,
    "training_ce_slope": 471.88114700965286
  },
  "R": {
    "norm": 1.1152698058649655,
    "training_ce_slope": -0.40617216872573236
  },
  "T": {
    "norm": 469.62151156315326,
    "training_ce_slope": 82.83182715781012
  },
  "even_norm": 882.4861554294579,
  "even_relative_ordered": 1494.9245658177254,
  "even_relative_frozen_error": 0.01756386201167726,
  "even_ce": 277.6130010312495
}
```

28161 quarter_step:
```json
{
  "P": {
    "norm": 0.0863732088414464,
    "training_ce_slope": 0.004733659840410812
  },
  "Q": {
    "norm": 0.07223217515800548,
    "training_ce_slope": 0.004052233024817551
  },
  "R": {
    "norm": 0.0837456197200467,
    "training_ce_slope": 0.001143041381015333
  },
  "T": {
    "norm": 0.08209811074894445,
    "training_ce_slope": 0.0011750038084685087
  },
  "even_norm": 0.010134014643025416,
  "even_relative_ordered": 0.11732821761465673,
  "even_relative_frozen_error": 0.04411139521526216,
  "even_ce": -0.0003247321940700426
}
```

28162 original:
```json
{
  "P": {
    "norm": 2.3309279632875186,
    "training_ce_slope": 0.3372958804477954
  },
  "Q": {
    "norm": 4656.815027014126,
    "training_ce_slope": -1533.0965646593797
  },
  "R": {
    "norm": 6.5751346901158625,
    "training_ce_slope": -2.8138610905265242
  },
  "T": {
    "norm": 859.1344951089962,
    "training_ce_slope": -311.53361403130043
  },
  "even_norm": 2350.2590434813083,
  "even_relative_ordered": 1008.2932979904386,
  "even_relative_frozen_error": 0.19131970023665407,
  "even_ce": -921.076806740301
}
```

28162 quarter_step:
```json
{
  "P": {
    "norm": 0.119421206505542,
    "training_ce_slope": 0.006973586824394624
  },
  "Q": {
    "norm": 0.11188494533988487,
    "training_ce_slope": 0.006585305641714195
  },
  "R": {
    "norm": 0.09033840699373849,
    "training_ce_slope": 0.004093997583656765
  },
  "T": {
    "norm": 0.08948026565615898,
    "training_ce_slope": 0.004106006557174351
  },
  "even_norm": 0.005235734148990044,
  "even_relative_ordered": 0.043842582923051175,
  "even_relative_frozen_error": 0.03530758596671753,
  "even_ce": -0.00018813610458141996
}
```

