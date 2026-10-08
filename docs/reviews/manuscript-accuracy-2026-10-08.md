# Independent manuscript accuracy pass — 8 October 2026

**Verdict:** the selected proof statements, heavy-ball correction and negative
practical results have appropriate scope. One material correction is needed:
the merged S2 crossing samples do not establish one continuously followed
full-state branch. The new lower S1R result must also be included as an unresolved
numerical gap. No theorem or displayed rigorous-bound blocker was found.

Reviewed manuscript snapshot:977514e4f53025f485bcd47ce947c9adef044843. Line numbers
below refer to that snapshot's `paper/gd-route/main.tex`; the companion receipt
binds exact file bytes. Lower-outcome dependency:175c6f785714b8e90c7e603e36eadee91ad4fd78.
Root is integrating that result and subsequent CIFAR/utility outcomes separately.
This reviewer did not edit the manuscript, regenerate figures, replay proofs,
run experiments or consume protected outcomes. Later manuscript edits need
integration review against these findings.

## 1. Correct the S2 continuous-branch implication

`tools/explore_a3_ns_curve.py:25–42` constructs the same initial system separately
for every output-rate task, runs its own bootstrap, follows hidden rate within
that slice, and saves crossing summaries without orbit states.
`tools/merge_ns_curves.py:21–30` selects by trace priority, deduplicates close
output coordinates and sorts rows; it performs no full-state pair matching.
`tools/explore_criticality.py:127–158` also bootstraps each output slice separately
before computing its coefficient. This is the same cross-slice identity issue
already correctly disclosed for S1R at main.tex:682–688. It does not invalidate
pointwise crossing calculations, coefficient signs or the eight tested onset
comparisons. It prevents an unqualified interpretation of the S2 overlay as
one followed full-state branch and of its three ordered sign transitions as
three zero crossings on that branch.

Suggested exact replacement for the second sentence at main.tex:285–288:

```latex
For the period-2 study, each fixed-output-rate slice is bootstrapped separately
and continued in hidden rate to its loss of stability. Several sets of crossing
summaries are merged into an output-ordered locus; full-state branch identity
between output slices has not been verified.
```

At main.tex:479–481, replace the opening through the semicolon with:

```latex
Independently initialized output-rate slices of S2-28011 produce a merged locus
of \nMergedPoints{} period-2 complex-pair crossings. Within each slice, the
orbit is continued in hidden rate to the unit-circle crossing;
```

Continue with the existing crossing-angle range. At main.tex:497–499, replace
the opening caption and panel-A description with:

```latex
\textbf{The first Lyapunov coefficient is negative at most sampled crossings,
and its sign predicts the tested local onset behavior.} A: $d$ at the merged,
output-ordered crossing samples (symmetric-log axis); dashed lines mark three
sign transitions between neighboring samples, whose full-state branch
association remains unverified;
```

At main.tex:512–514, replace the sign-change/Chenciner sentences with:

```latex
The ordered samples have \nChenciner{} sign transitions, near
$\log_{10}\eta_o$ = \tChenciner{}. These nominate locations for a
full-state search for generalized Neimark--Sacker (Chenciner) points;
association of the neighboring samples to one branch and the additional
nondegeneracy conditions have not been established.
```

Retain the existing pointwise counts and dynamical-check results. Main.tex:518
can say “\nCritSOneRSuper{} of \nCritSOneR{} sampled S1R coefficients are
negative” to make the same conditional criticality scope explicit.

Follow-through needed for consistency:

- Main.tex:125–127: replace “crosses the unit circle along a curve” with
  “crosses the unit circle in independently initialized slices”; keep the
  local onset evidence and the global negative result.
- Main.tex:155: replace “a traced curve of complex-pair crossings” with
  “a merged locus of slice-wise complex-pair crossings.”
- Main.tex:467–474: call the black overlay a merged crossing locus; its
  connecting line is a guide between sampled crossings, not a followed branch.
  Call interpolated rational-angle locations provisional resonance candidates.
- Main.tex:875–878: replace “the continuity of the rotation number with the
  crossing angle in two maps” with “the agreement of sampled rotation numbers
  with the corresponding crossing angles in two maps.” This leaves the measured
  errors intact while avoiding a continuity theorem from finite samples.
- Main.tex:54–55 can read “Exploratory calculations identify a local organizing
  mechanism: at selected period-2 orbits, a complex pair of multipliers crosses
  the unit circle, with an angle that predicts the ensuing oscillation frequency.”
  The existing abstract already avoids asserting a global route.

Carry the wording into figure source/alt text and metadata, not just prose:
`f2_rotation_map.py:108,117–121`, `f7_criticality.py:83–84,136–143`, and
`build_numbers.py:486–493`. Suggested panel title: “Criticality at sampled
crossings; dashed: neighboring sign transitions.” Suggested computation label:
“sign transitions in output-ordered coefficient samples; branch association
and generalized-NS nondegeneracy unverified.” Keep the three stored midpoint
values and all historical evidence unchanged. The existing numeric key
`sign_changes` can remain as a historical schema name with this interpretation.

## 2. Crossing terminology in figure accessibility text

The f2 and f7 alt text says the orbit “acquires complex multipliers.” The
computed event is a complex pair **crossing the unit circle**; the pair can
already be complex and damped before that event (the manuscript's AR discussion
uses precisely this case). Replace that phrase in
`f2_rotation_map.py:117–118` and `f7_criticality.py:136–138` with “has a complex
pair crossing the unit circle at the sampled settings.” Regenerate the figure
receipts from the unchanged data. This is a terminology correction, not a new
measurement.

## 3. Integrate the lower-family negative result without upgrading it

The reviewed snapshot stops at the upper-bootstrap association (main.tex:700–711).
The separate lower outcome is now available and independently audited. A
compact insertion immediately afterward could read:

```latex
A further continuation toward the lower bootstrap controls completes 54 of 75
nominal hidden-rate segments, with 56 accepted full-pair links. Its last
witnessed primitive pair, at $\log_{10}\eta_h=0.992$, is numerically unstable.
The next target, 0.990, fails at five successive proposal levels, so the lower
endpoint at 0.950 is not reached and no common-control association is obtained.
An independent saved-data audit reproduces the retained qualifications and
witness decisions, with independent analytic derivatives at five prescribed
pairs. This numerical gap proves neither a fold nor global disconnection.
```

Use generated macros bound to both producer and independent outcome receipts,
not the literal illustrative numbers above. Cite the unchanged producer report
and `docs/reviews/s1r-lower-independent-outcome.md`. Distinguish54 complete macro
segments from56 accepted edges: one macro used smaller successful links. The
terminal historical `halvings=5` counts five failed proposal levels (initial
plus four applied halvings), not five applied halvings. No sixth proposal was
run. The original creation-time pending-review receipt must remain unchanged.

No abstract proof or coexistence claim follows from this result. The contribution
at main.tex:160 can add “an unresolved lower-family comparison.” The exact-real
72D point certificate remains valid and separate from this numerical path.

## 4. Smaller scope clarifications

These are secondary to the branch-association correction:

- Main.tex:285–286 and590–593 classify boundaries by crossing multipliers.
  Say “fold candidate ($+1$), flip candidate ($-1$), or complex-pair candidate”
  once in Methods, consistently with main.tex:189–190. Multiplier type alone
  does not establish all fold/flip nondegeneracy conditions.
- Main.tex:520–524 gives evidence for a basin-accessibility explanation, but
  “the cause is not the bifurcation” is unnecessarily absolute. Suggested last
  sentence: “These comparisons show that the fixed-start divergent arc cannot
  be identified with local subcriticality alone; basin accessibility from the
  shared initialization matters.” Preserve the stated below-boundary and
  near-orbit observations.
- Main.tex:717 describes a “smooth invariant circle” for the digit network.
  “A smooth projected loop consistent with an invariant circle” matches the
  finite-time FTLE, rotation and projection evidence more precisely. This
  remains different from the scalar exact-real local theorem.
- Evidence-table lines90–91 could mention the two negative transformer
  prediction pilots and the checkpointed CIFAR corrector failure. This is a
  completeness update; detailed sections already state their limitations.

## Checks that pass and remaining scientific gaps

The 72D certificate passage (383–396) correctly separates exact-dyadic real-map
existence/primitivity/attraction from binary64 trajectory periodicity, numerical
seed proximity, branch identity and parameter continuation. It correctly states
that the higher-precision/differentiation cross-checks above concern S2; the
S1R audit shares the reviewed proof checker. The displayed2048-power bound0.556
is rounded upward from the exact stored bound.

The scalar theorem (401–437) retains exact dyadic coefficients, actual radius
2^-40+2^-69, full planar spectrum, nonresonance through order4, implicit orbit
branch, positive **modulus** slope and negative normalized cubic coefficient.
Its two exchanged local components under F, qualitative sufficiently-small
positive detuning, lack of guaranteed irrationality and provenance limitation
are correctly stated. All four displayed interval endpoints enclose the exact
saved balls by rational comparison. The two S2 power64 bounds0.056/0.108 are
also conservative. The similarity corollary has the correct k² slope and k^-2
coefficient scaling; it is explicitly one dimensionless case, not replication.

The heavy-ball polynomial and modulus sqrt(beta) argument (220–230) are correct
for constant positive rate and fixed0<beta<1. The text properly refuses to infer
an Adam fixed-point NS mechanism from nonsymmetry. The frozen-prefix logit-margin
argument (915–918) is correct and does not establish stability of an entire
generated sequence or chaos at inference.

Practical claims remain appropriately negative: matched-history B3 is a
nonlinear representation of B2's information; both broader-data gates fail,
with the stated tuning-selected ordinary comparators rather than post hoc
selection on test error. The displayed paired nMSE values match the retained
independent audit. Probe improvements are descriptive, one-pool comparisons;
probe forward evaluations were not independently replayed. The rate-ramp
comparison identifies neither a dose-matched superior controller nor a
geometry-specific benefit. No new LLM, stochastic-training or inference benefit
is implied by the abstract or discussion.

Current precise citation contexts agree with the retained primary-source
content reviews for the core EoS, game/heavy-ball, boundary, adaptive-optimizer
and circle references. This pass checked that follow-through; it did not
independently re-audit every external proof. Keep the ledger's actual open items:
Arnold1965 remains unchecked; Jensen1983/Aronson1982 have only abstract-level
content checks; some final publisher/preprint comparisons and detailed book
formula checks remain open. Do not label the whole bibliography fully verified.
No new browsing, paid consultation or external message was necessary.

Scientific gaps remain: cross-output full-state association before promoting
S2 sign transitions; generalized-NS degeneracy checks; a connected global
resonance/window path; a specified parameter-range network bifurcation proof;
and prospective geometry-specific utility against strong ordinary controls at
comparable cost. A failed continuation does not resolve topology, and a single
point certificate does not supply any of these missing links.

## Root working-diff follow-up before review commit

Root applied the main locus/adjacent-row clarification in the uncommitted
manuscript, f2/f7 labels/alt text and number-metadata descriptions after receiving
finding1. This reviewer inspected that four-file diff: the Methods, main
crossing and criticality passages, figure captions and Limitations now preserve
the sampled pointwise interpretation. No scientific calculation or measured
value changed in the inspected diff. Generated figures/numbers were not yet
regenerated, so this is not a build signoff.

Remaining follow-through at that inspection: introduction125, contribution155
and discussion876 still had the older continuous-curve wording, and both f2/f7
alt texts still said “acquire(s) complex multipliers” instead of a unit-circle
crossing. Exact replacements are above and were sent to root. The independent
review receipt also binds the inspected working-file bytes separately from the
original snapshot. Root owns applying remaining edits, adding new outcomes and
compiling; this report does not approve uninspected future edits.
