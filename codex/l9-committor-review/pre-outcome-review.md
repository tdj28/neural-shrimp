# L9 committor: independent pre-outcome source review

Date: 13 October 2026, before any L9 outcome access by this reviewer.
This is the original-source disposition; see the separate amended-source review
for subsequent fixes and the producer's disclosed limited progress exposure.
Reviewer: Codex `boundary_resume`. Requested analysis source and run card:
`2deec409779fa1a92e34555f4d0b3928559f1823`.

**Behavioral conclusion permitted by the design:** repeated continuations can
estimate short-horizon hostile-reply risk at selected dialogue states and compare
that estimate with fixed detector scores. They do not, at K=32, establish a bound
on every detector or identify a memoryless-versus-approach mechanism. No L9 result
is evaluated or endorsed here.

Scope: I read the pinned `select`, `rollout`, and `analyze` source, the L9 run card
(`docs/research-log.md:21919–21924`), the 12:50Z request, and the six source
dependencies in `source-manifest.json`. Exports were provided by the sole Git
owner. Every exported byte count and SHA-256 was checked using only standard
library reads. I did not import producer code, read original trajectories or L9
rollout/outcome files, execute analyses/models, inspect jobs/pods, or use Git.
The argument below is source inspection and mathematics, not an executed data
audit. Line references to `p1_committor.py` mean the pinned `tools/` source.
The run card's rollout commit `8435fb793abcad42f7070b9ca8b799ed1e1657ad`
has byte-identical blobs for all seven reviewed Python sources, independently
exported by the Git owner in `rollout-source-comparison.json`. Thus that specific
commit-name difference is not a source-code discrepancy.

## Findings that must change interpretation or analysis admission

### 1. Estimated AUC is not a detectability ceiling

**References:** `p1_committor.py:4–10,239–258,374–398`; run card `21921–21924`;
request `comms/claude-to-codex.md:224–229`.

For a declared target distribution of at-risk windows, let X be the information
available before the next reply and Y the fixed horizon event. If
q(X)=P(Y=1|X), then, writing pi=P(Y=1),

`dP(X|Y=1)/dP(X|Y=0) = q(X)/(1-q(X)) * (1-pi)/pi`.

Thus thresholds on the **true** q, with suitable randomization at ties, achieve
the population Bayes ROC envelope among tests measurable with respect to X.
This is not an inequality between empirical AUCs on a particular realized
sample. For example, two states with true risks .8 and .2 can realize labels
0 and 1, respectively, with probability .04 under independent outcomes. The
true-risk ordering then has empirical AUC 0 and the reverse ordering AUC 1.

Moreover, K independent rollout indicators give a binomial estimate with
conditional variance q(1-q)/K, at most 1/128 for K=32 (SD about .0884), on the
33-point grid {0,1/32,...,1}. Ties and rank reversals need not vary linearly in
1/K. The unconstrained fit through 8,16,32 has no established universal
extrapolation law or bound; its estimates can exceed [0,1] or have the opposite
slope. The subset estimates also share rollout data. Repeating this fit inside
a bootstrap does not validate the extrapolation model or turn it into an upper
confidence bound for Bayes performance.

**Minimal change:** call q32 an estimated rollout-risk reference and its AUC an
empirical comparison. Label A_inf a heuristic exploratory extrapolation, retain
all three AUCs/slope, and remove “ceiling,” “share of attainable warning,” and
“nothing left to detect” as established conclusions. Do not infer intrinsic
unpredictability from a small detector–q32 gap. The ratio at line 398 is also
undefined at A_inf=.5 and can be unstable or misleading near it; bootstrap
line 392 selectively omits nonpositive denominators. Report these omissions,
not a ratio interpreted as a fraction of a proven maximum.

The X/policy contract must include the known steering dose, prompt/window,
speaker phase, generation and event rules, and any other relevant past
information available to the compared detector; exclude unobserved future RNG
state. The simulator also has model access and 32 fresh continuations per state.
Population oracle optimality is not a claim of equal deployment computation or
an online benefit. Historical labels/onsets used to select states are not
permitted predictor inputs.

### 2. The bootstrap splits dialogues across outcome strata

**References:** `56–97,261–270,356–369,385–395`.

The selection unit is a row, sampled without replacement within its h=3 outcome
stratum. A single dialogue can contribute an early negative row and a later
positive row. The bootstrap builds separate dialogue lists inside those two
strata and draws them independently. Consequently the positive and negative
parts of one dialogue can receive different multiplicities. This is not a
whole-dialogue cluster bootstrap. For two dialogues each represented in both
strata, the same dialogue's multiplicities differ with probability 5/8 under
this construction. Cross-stratum dependence is lost; the sign of the resulting
CI error cannot be inferred from source alone.

Target stratification and dependence are different issues: inverse selection
weights address the row sample's case/control composition, but do not restore
that covariance. Conversely, merely resampling whole dialogues does not by
itself exactly reproduce the fixed-size, without-replacement row design.

**Minimal change:** declare the uncertainty target before viewing results
(conditional finite historical population versus a population of new
dialogues). For dialogue-level uncertainty, share a resampling/multiplier weight
across every selected row of each dialogue, optionally stratifying by a
dialogue-constant arm, and retain inclusion weights. If conditioning on fixed
outcome-stratum totals, specify the corresponding calibrated/two-stage design.
Record this as a pre-outcome amendment, preserve the original computation, and
record every undefined bootstrap draw instead of silently dropping it through
`nanpercentile`. Distinguish uncertainty of the finite-K reference from
uncertainty about the unobserved true q or an extrapolated Bayes AUC.

### 3. Primary analysis currently accepts unverified or partial rollout sets

**References:** `128,147–150,174–175,203–210,230–236,273–281,338–355`.

The loader concatenates every matching block and converts keys to a dictionary;
duplicates overwrite earlier values. Analysis then silently drops states with
missing keys. Neither step verifies the planned complete set, K=32/H=3, exact
state selection hash, finiteness/range of scores, or block provenance. A NaN
toxic score becomes a non-hostile Boolean at line 233. Existing block files are
skipped on resume without matching their parameters or state keys. The final
`rollout.json` records a states hash, but `analyze` never reads it. Original
sample/hostility hashes stored by `select` are also not rechecked; test features
and training arrays have no authentication in this analysis path.

**Minimal change before primary analysis:** authenticate the frozen selection,
actual rollout source/settings and asset identities; require exactly one block
entry per planned key, the full planned state set and counts, score shape
32x3, finite scores in [0,1], text/score association, positive selection weights,
and exact test/train arm identities with disjoint dialogues. Reject unknown,
duplicate, missing, stale or malformed data. Preserve any partial run separately.
Randomizing state order does not license arbitrary incomplete data to inherit
the full-design claim, and provides no repair for duplicate or stale blocks.
Do not invent missing provenance; report it as unavailable if it was not saved.

### 4. The declared fidelity restriction is not enforced

**References:** `358–374,436–453,490–491`; run card `21922`.

The run card says an interval excluding zero makes the proposed ceiling
uninterpretable and requests 2000 CPU re-scores. Code prints calibration and
continues unconditionally; re-scoring defaults to zero and has no explicit
admission check. Enforce or explicitly record the **already declared**
calibration restriction; this review adds no new outcome-dependent threshold.
Keep downstream descriptive quantities if useful but mark the declared
interpretation gate failed. Record whether the prescribed 2000 checks were
actually performed and every disagreement.

Even a passing aggregate mean check is not proof of conditional-process
equivalence: opposite state-specific errors can cancel. An excluding interval
is a statistical warning under the chosen CI procedure, not a logically certain
demonstration of process mismatch. The current rescore sampler chooses a block
uniformly then a reply within it, so the shorter last block is overweighted
(`443–447`); call it that sampling design or weight/sample replies uniformly.

### 5. Onset-aligned profiles do not classify the mechanism

**References:** `84–96,402–419`; run card `21922`.

The profile conditions on eventual capture at onset>=11, aligns backward from
that future onset, and contrasts it with dialogues selected never to capture
over the observed horizon. This is a useful retrospective precursor description.
It is not a prospective sample of risk states and not a controlled test of a
memoryless hazard. Time-dependent hazards, risk heterogeneity, survival/future
selection, and changing state mixtures can affect these curves without an
attractor-approach mechanism. Flat estimated curves do not prove memorylessness;
rising curves describe conditional-risk variation, not a unique cause. Neither
“flat” nor the multi-offset shape decision has an executable threshold/test.

**Minimal change:** label the profile descriptive and conditional on the stated
selection. Require complete matched pairs at all offsets rather than silently
using different available sets. Resample entire paired trajectories jointly
across offsets for a shape comparison; per-offset intervals are pointwise, not
a simultaneous band. A later mechanistic test needs a specified memoryless
null, survival/selection-matched comparison and separately declared test.

## Source checks that are sound within their stated scope

- **Event/censoring:** `p1_detection.py:36,41–67` and `p1_committor.py:100–104`
  implement first hostile score strictly greater than .5, no-event sentinel
  handling, at-risk rows before onset, and complete horizons. L9 is the primary
  first-hostile event, not the two-consecutive secondary event. Its weighted
  h=1/2 comparisons use the h=3-eligible row population: they omit the extra late
  rows available to a native h=1/2 analysis. State that target explicitly.
- **Weights:** `79–83,215–227` implement inverse row-stratum sampling weights and
  half-credit AUC ties. At h=3 the weight is constant within each class and
  cancels from AUC; weighting remains relevant for prevalence and shorter-horizon
  labels on this selected population. Weighted AUC is a ratio estimator, not an
  assertion of exact finite-sample unbiasedness under the without-replacement
  design. The claimed population counts still require a separate data audit.
- **Transcript/seed logic:** `134–137,177–195` continues through reply t, then
  advances the speaker correctly: `(len(transcript)-1)%2` equals the next
  original turn parity (`explore_llm_msm.py:75–96`,
  `explore_llm_attractors.py:43–56`). Default block/turn/chunk seeds are distinct
  within this declared run and do not intentionally copy the realized future.
  `generate_batch` resets a generator once per batch and samples separate rows
  (`explore_llm_phase_map.py:148–161`); the K copies are not each reset to the
  same seed. Independence is the usual PRNG sampling assumption, not a theorem
  from distinct integers. Actual original seed ranges and runtime identities
  were not read or attested here.
- **State timing and fitting:** `284–288` uses `pre_reply[t+1]`, whose recording
  hook is before sampled next-reply tokens (`explore_llm_phase_map.py:123–130`).
  `p1_detection_confirm.Frozen:33–57` fits PCA/scalers/logistic only on supplied
  training data; L9's intended L7/L8 separation has no evident transform leakage.
  It refits the frozen recipe, rather than authenticating saved fitted objects.
  The classifier is pinned to toxic-bert revision
  `4d6c22e74ba2fdd26bc4f7238f50766b045a0d94` and verifies label order
  (`hostility_score.py:29–31,50–73`). These source properties do not authenticate
  the actual data/assets. The generation model/tokenizer are loaded without an
  explicit revision (`explore_llm_attractors.py:74–76`), so the actual cached
  revision and vector bytes must be recorded separately.
- **Thresholds:** the frozen detector thresholds use training negatives only
  (`p1_detection_confirm.py:46–49`), but L9 reports AUC, not an operating point.
  It cannot support a new fixed-FPR sensitivity claim without an explicitly
  declared threshold analysis. Report converged-fit diagnostics/warnings and
  finite scores; the present calls do not retain those diagnostics.
- **Soft-target probe:** folds hold out whole dialogues, and transforms fit
  training folds (`308–331`). There is no obvious same-dialogue fit/test leak.
  However, both fits use an outcome-selected mixture, without selection weights,
  and soft-target fitting uses all training rows while outcome fitting excludes
  censored labels. A difference therefore is not isolated evidence for soft
  targets alone. Keep exploratory, state that support difference, and predeclare
  a same-row comparison if that narrower question is desired. Record all folds,
  class degeneracies, fit warnings and iterations; do not tune on held-out L8.

These are previously studied L8 dialogues and labels. Fresh continuation draws
add Monte Carlo information, not a new held-out set of original dialogues.
The run card correctly calls L9 exploratory; retain that label and the narrow
classifier-defined hostility proxy rather than generalizing to all misalignment.

## Release recommendation

Do not promote the proposed empirical ceiling or mechanism classification.
Before reading L9 outcomes, disclose a compact amendment correcting those
interpretations, specifying uncertainty, enforcing complete provenance/data
admission, and retaining the original fidelity rule. No new model acquisition
is required to correct the wording or source gates. Any missing evidence must
remain a qualification, not be replaced by a successful-looking analysis.

This review does not evaluate a single L9 result. No scientific test was run;
source hashes and the mathematical counterexamples above are the review evidence.
The manifest binds exact source files so later edits cannot inherit this review
without a delta check.
