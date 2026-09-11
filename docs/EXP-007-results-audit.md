# EXP-007 independent saved-outcome audit

Date: 2026-09-11. Source: `ac098c3e3833daaf27c7dc2869d45e8b7085f948`.
Attempt: `EXP-007/attempt-001`. This audit ran independently from the
experiment's analysis helper. It imports no experiment module, invokes no
integrator and modifies no attempt file. It reconstructs the saved decisions
and outcomes rather than treating the reported summary as evidence of itself.

## Outcome and scientific interpretation

**The frozen hybrid has not established an advantage over the strong
comparators on these new interventions.** All five primary pooled paired
95% intervals include zero at both numerical steps. The hybrid's fine-step
mean tail RMSE is 0.06237278681, compared with 0.09859456705 for sign-probed
local sensitivity, 0.05109471923 for tangent/axes, 0.07417114465 for uniform-16,
and 0.04924624261 for augmented search. Lower is better. These are the
predeclared equal-anchor, equal-phase, equal-dose means, not chosen examples.

There is a clear dose-dependent manifestation in actual three-neuron
activation waveforms. At the two smallest doses the local policy, which the
hybrid copies, performs better than tangent/axes, uniform and augmented
search; all corresponding descriptive intervals are positive. Geometry alone
has substantially larger errors there. At radius 0.004 the frozen hybrid
retains local sensitivity, but its mean error is 0.13262982028, compared with
0.02712988834 for tangent/axes. At 0.016 the atlas/hybrid mean is 0.11544843321
versus 0.26033555417 for local, but the paired benefit interval
[-0.04552387742, 0.35319372521] includes zero and only 6/12 anchors favor the
hybrid. The augmented search mean is lower still at 0.07664364844. This does
not replicate a reliable large-dose geometric advantage under the new
stronger test. It also does not prove geometry useless: uncertainty remains
large and this conditional assay is small.

A crucial attribution check found **exact equality of the tangent and signed
axes actions in all 192 conditions**. Their selected preview arrays are equal
in all 192 cases; their target state arrays and complete tail-metric objects
are equal in all 384 comparisons across both steps. They choose +gamma in
168 cases and -gamma in 24. Their performance here therefore supplies no
independent evidence that an atlas tangent is needed: the map-free signed-axis
search reaches exactly the same actions. This is an empirical identity for
this slice and these decisions, not an identity of the two algorithms on
arbitrary maps.

## Independently reconstructed primary numerical results

Each nonzero cell averages four phases at each of 12 eligible anchors. The
primary score is activation RMSE on elapsed samples 75.1 through 300, fitting
one global phase on those 2,250 samples. No frequency fitting or per-window
phase refitting is used. Full-window scores are separate secondary outcomes.
The figures below are rounded only for presentation; the audit JSON retains
full precision.

| Radius | Atlas | Local | Tangent / axes | Uniform-16 | Augmented-21 | Hybrid |
|---|---:|---:|---:|---:|---:|---:|
| 0.00025 | 0.013148576 | 0.000243803 | 0.002412151 | 0.001439584 | 0.000580387 | 0.000243803 |
| 0.001 | 0.056555159 | 0.001169091 | 0.008732162 | 0.005639088 | 0.002351232 | 0.001169091 |
| 0.004 | 0.135219172 | 0.132629820 | 0.027129888 | 0.107517413 | 0.117409703 | 0.132629820 |
| 0.016 | 0.115448433 | 0.260335554 | 0.166104676 | 0.182088495 | 0.076643648 | 0.115448433 |
| pooled | 0.080092835 | 0.098594567 | 0.051094719 | 0.074171145 | 0.049246243 | 0.062372787 |

Benefit means comparator RMSE minus hybrid RMSE; positive values favor the
hybrid. Intervals use 10,000 resamples of whole anchors with seed 20260912,
retaining paired phases and doses. These are exploratory percentile intervals,
not multiplicity-adjusted tests or population-generalization guarantees.

| Radius | Local benefit [95% interval] | Tangent / axes benefit [95% interval] | Uniform-16 benefit [95% interval] | Augmented-21 benefit [95% interval] |
|---|---|---|---|---|
| 0.00025 | 0.000000000 [0.000000000, 0.000000000] | 0.002168348 [0.001043765, 0.003428692] | 0.001195781 [0.000559336, 0.001942176] | 0.000336584 [0.000085627, 0.000630517] |
| 0.001 | 0.000000000 [0.000000000, 0.000000000] | 0.007563071 [0.003688213, 0.011819618] | 0.004469997 [0.002064933, 0.007362742] | 0.001182141 [0.000345671, 0.002091481] |
| 0.004 | 0.000000000 [0.000000000, 0.000000000] | -0.105499932 [-0.229253428, 0.010965284] | -0.025112408 [-0.113618133, 0.035598804] | -0.015220117 [-0.064110502, 0.025316786] |
| 0.016 | 0.144887121 [-0.045523877, 0.353193725] | 0.050656242 [-0.043143453, 0.153512339] | 0.066640061 [-0.042758755, 0.201204879] | -0.038804785 [-0.097361832, 0.031162564] |
| pooled | 0.036221780 [-0.011380969, 0.088298431] | -0.011278068 [-0.050719712, 0.028090938] | 0.011798358 [-0.028639921, 0.052582139] | -0.013126544 [-0.033382635, 0.008182230] |

The local/hybrid zero contrasts at the three smaller doses are structural:
the hybrid literally copies the local action. Similarly, atlas/hybrid are
identical at the largest dose. Neither identity is independent validation.
The additional pooled atlas-minus-hybrid contrast is 0.01772004827
[-0.01231752507, 0.04669011222], also unresolved.

## Numerical, source and census verification

- All 7,149 receipt artifact hashes and sizes pass, totaling 374,438,288 bytes.
  All 185 old-input artifact hashes pass. The attempt manifest is byte-for-byte
  identical to the plan in the frozen source commit. The artifact census has
  no unexplained extra non-build files.
- All 12 planned anchors are eligible; both baseline period classifications
  are nine and their saved independent certificates report verified,
  attracting cycles. No anchor was replaced. The baseline orbit has the
  prescribed 2,700 endpoint-excluded states. Fresh initial states and epochs
  match the four prescribed indices exactly. This audit checks saved
  certificate consistency, not a new independent shooting solve.
- All 4,032 previews and 2,784 targets completed. All expected unique target
  IDs, 192 decision contexts and 96 zero-dose shams are present. There are no
  failed, timed-out, selector-failed or baseline-ineligible records. The
  preview census is independently reconstructed, including its wall-time sum.
- All 6,816 preview/target raw traces have the exact saved starting states,
  time arrays, parameter updates and integration steps. Native direction norms
  differ from one by at most 2.22e-16. Recomputed tail sensitivity Grams and
  local directions agree with their saved values; no initial-state sensitivity
  or preview samples are added to the tail Gram.
- Every selected candidate is independently recomputed from its declared
  preview subset with the fixed lowest-index tie rule. All policy choices
  match. The hybrid's radius switch matches the frozen plan. The augmented
  search selects a local candidate 107 times, an atlas candidate 18 times and
  a uniform candidate 67 times. Its access to geometry is therefore explicit;
  it is not the map-free competitor.
- Independent periodic interpolation at each recorded single global phase
  reconstructs 9,600 metric blocks and 504,320 aggregate, per-neuron and
  per-window RMS values. The maximum discrepancy is 4.44e-16. This audit uses
  the unchanged rtol=1e-12, atol=1e-14 comparison and does not refit an
  optimizer. Every scalar preview score matches its saved metric. No saved
  reference-spectrum cadence flag is raised; the diagnostic cannot exclude
  aliasing in the reference itself.
- Independent phase/anchor/dose aggregation and anchor-cluster bootstrap
  reproduce every saved primary mean, contrast and interval. Zero means and
  maxima use all required shams. There is no success-only deletion.

The fine-step zero-sham tail floor is mean 0.00006534566104 and maximum
0.00006883853467 (48/48); the coarse-step values are 0.00006535291262 and
0.00006887144255 (48/48). Across all 1,392 matched target pairs, absolute
coarse/fine tail-RMSE differences have median 0.00000013030, mean 0.00402639515,
95th percentile 0.01456181788 and maximum 0.23486410947. Thus individual
trajectories are not uniformly converged. Both steps give the same main
aggregate conclusion; agreement of that conclusion must not be phrased as
uniform trajectory convergence. The primary fine step was frozen in advance.

Coarse-step pooled means are atlas 0.07972186581, local 0.10153440396,
tangent/axes 0.05296038415, uniform 0.07354277283, augmented search
0.04912615284 and hybrid 0.06313165734. All five primary pooled intervals
again cross zero. At the largest dose the coarse local-minus-hybrid benefit
is 0.15361098649 [-0.03813577056, 0.36245436759].

## Frozen illustration, cost and claim limits

The protocol's illustration is the first eligible anchor, `holdout-00`, phase
337, radius 0.016, dt 0.005. Its tail RMSE is 0.05966803102 for hybrid/atlas,
0.03776337582 for local, 0.66883677875 for tangent/axes, and 0.58975318592 for
both uniform and augmented search. The local action performs better than the
hybrid in this unselected example. A different visually persuasive anchor
must not replace it on the basis of these outcomes.

The collection receipt reports 137.50909017 seconds before final verification.
This includes the shared experiment collection, not separately measured
per-policy execution. Forty-eight local sensitivity solves use 23.51814112
solver wall seconds and 2,569,452 function evaluations. The 4,032 preview calls
sum to 161.84820580 individual call-wall seconds; the 2,784 target calls sum
to 228.62682979. Concurrent-call sums are not elapsed runtime and cannot be
added to the receipt wall time. Paid compute and paid review are both zero.

Counterfactual preview budgets across all 192 eligible conditions are atlas
0, local 384, tangent 384, axes 768, uniform 3,072, augmented search 4,032 and
hybrid 288. Local and tangent have equal two-preview budgets but different
upfront information costs. Atlas/tangent/hybrid/augmented search rely on the
previously built atlas; local/hybrid/augmented search require sensitivities.
The shared cached pool does not make these access costs free, and this run
establishes neither equal-total-cost superiority nor a measured net speedup.

These are new starting anchors and interventions conditional on an already
observed atlas. The halo excludes prior starting anchors, not every previous
intervention destination. Some new anchors are near neighbors across rank
thirds. Four phases per anchor and repeated actions are not independent
replicates. The 75-unit simulator preview and the later scored samples are
disjoint but dynamically dependent. More preview candidates can overfit that
short horizon and have worse later error; the uniform or augmented score
need not improve monotonically on the held-out tail. Their selection is
valid because it never sees tail outcomes.

The endpoint measures preservation of a chosen reference activation waveform
under one fixed weight change in this three-neuron model. It does not measure
trained-task accuracy, energy savings, a repeated controller or a general
learning algorithm. EXP-007 does not perform long-time period classification
at perturbed targets, so these errors do not by themselves establish future
period-nine retention or loss. Any switch, dose or preview horizon redesigned
using these results requires a newly frozen prospective assessment.

**Audit disposition:** saved evidence and derived results pass this independent
audit. The scientific result is bounded and mixed; the broad utility claim
remains unestablished. The runner's own full verification subsequently completed with exit status zero.
The copied repository evidence also passes this separate audit.

## Reproduction records

The retained standalone audit is `tools/audit_hopfield_holdout.py`; its full
machine output is `docs/EXP-007-independent-check.json`. The default input is
this repository's `evidence/EXP-007/attempt-001`; alternate paths can be supplied
with `--repo` and `--attempt`. `--out` is optional and may not point inside the
immutable attempt. For example:

```sh
python tools/audit_hopfield_holdout.py --out /tmp/EXP-007-independent-check.json
```

No experiment analysis or selector implementation is imported. The source
commit and manifest bind the experiment. The receipt SHA-256 is
`5a944722af26b9406bf2ba881067305491a115c41fe44a282ad2e9cb72282d23`.
The following hashes bind this separate audit packet; re-running changes the
audit's wall-time metadata without changing the reconstructed scientific
results.

- `tools/audit_hopfield_holdout.py` SHA-256 `fccd4e36cfec0fcdaf2d9517cf4574db151de1f407ae0f8b4d6cad7ca13acff4`.
- `docs/EXP-007-independent-check.json` SHA-256 `64c4b058211a561d3ccbb0ebabe687558bc6a4f6a65258a481d55733deee7427`.
