GO WITH FIXES — article bb42a82; make the nine must-fix corrections before submission.
17 findings: 9 must-fix, 7 should-fix, 1 note; explicitly partial checks are listed below.
A3: all 225 CPU/GPU cells agree at the paper's precision; maximum leading-multiplier difference 6.554×10^-12, with two temperature rows differing by one ULP.

# Task40 independent article and evidence review

Article reviewed: `bb42a82189da91c37f50804c077832df9e2237ae`, against Task39's
`4ca5b5e35bc766581a5e2d326858fcd0edc312bb`. Research reference:
`c567545ca93f90324dae75be2c333c8ac452dab5`. All article locations below refer
to `/data2/chaos-journal-neural-shrimp/main.tex` at **bb42a82**, not the
longer research report or subsequent author edits. The frozen article source
is 164,521 bytes, SHA256
`b0ecb1d8b74d19bb0b1a6b87c402846679f12b50346d135f9a0beecf45d562f1`.

The human directed that the arXiv review be delivered before the Zenodo
package audit. This first release therefore gives the article verdict now;
package contents remain pending in the coverage ledger. Present-tense public
availability must nevertheless be truthful in the submitted article.
No article, producer source, existing evidence, repository visibility or
publication state was changed by this audit.

The evidence supports the paper's principal numerical account after the
corrections below. In particular, an independent SVD/subspace-overlap
implementation reproduces the trace saddle and zero-set statistics, without
importing or rerunning the producer analysis. The main new findings concern
quantifiers, numerical bounds, and distinctions between numerical evidence
and stronger claims. This is not an unconditional approval of the frozen
PDF, a new proof of chaos, or verification of every original model execution.

## Findings

### F1. Natural-passage errors need their statistic and channel definition [must-fix]

Where: Sec. V, “Passages and the window law,” lines 927–929; Table IV,
lines 948–960; compare Fig. 14(d), lines 817–818.

Claim: “From each measured entry point the normal form predicts the length
of the laminar phase that follows within 4–8% for eps <= 10^-4.”

Evidence: `fold_saved_audit.py` independently reduces
`passage-natural-p2-t-0.115/{passage.json,u.npy,transverse.npy}`. The original
statistics use **both |u|<3U and transverse distance<0.05**, keeping complete
phases only. Across the four offsets, median signed errors
`(observed-predicted)/predicted` are −4.8782%, −4.0816%, −7.4074% and
−7.6923%; largest individual absolute errors are 12.5%, 64.94%, 38.64%
and 70.59%. The figure instead uses |u|<U alone; under that definition the
fourth median is −11.11%. Simply adding “median” without identifying the
channel is insufficient. See [fold-review.md](fold-review.md), S1, and
`fold-evidence.json`, `natural`.

Correct value / wording: “For complete phases defined by |u|<3U and
transverse distance below 0.05, the median signed forecast errors at the
four offsets up to 10^-4 range from −7.7% to −4.1%; individual errors reach
71%. Their mean lengths are approximately 0.60–0.67 of the full passage
through this channel.” State the forecast denominator and exclude censored
phases explicitly. Table IV must identify the same definition. Fig. 14(d)'s
separate definition can remain. The independently checked prospective
42/48 result is unaffected.

### F2. The cusp bracket range omits small endpoints [must-fix]

Where: Sec. VII, cusp unfolding, lines 1379–1380.

Claim: the quadratic coefficient falls “from between 2.4 and 39.2 at the
bracketing curve points.”

Evidence: All twelve initial endpoints in the six `locate.json` records
were checked by `raw_evidence_audit.py`. Their absolute extrema are
**0.17047709685584778 and 39.1082399960387**. Taking only the larger
magnitude in each bracket gives a different statistic, with minimum
2.459650167855898. All six final values satisfy the quoted 9.5×10^-5
bound: maximum 9.477318099015974×10^-5. See
[raw-review.md](raw-review.md), A6, and `raw-evidence.json`, `cusps`.

Correct value / wording: “Across the twelve bracketing curve points,
the quadratic coefficient has magnitude approximately 0.17–39.11; at
each located cusp it falls to at most 9.5×10^-5.”

### F3. The abstract identifies two different feedback operators [must-fix]

Where: Abstract, lines 43–44.

Claim: feedback “over the full vocabulary, as in Soft Thinking.”

Evidence: The published Soft Thinking method filters and renormalizes its
distribution. The primary NeurIPS paper and publisher metadata were read;
the introduction already makes the distinction correctly. The accepted
external-review summary says this was fixed, but the abstract retains the
identification. See [citations-review.md](citations-review.md), F1, for
the exact primary links and passages.

Correct value / wording: Delete “as in Soft Thinking” here, or say “a
smooth full-distribution variant of Soft Thinking.” Filtering can alter
the map's smoothness and is not an incidental implementation detail.

### F4. The introduction overgeneralizes synchronization thresholds [must-fix]

Where: Introduction, lines 167–168.

Claim: copies “synchronize near the coupling that the theory of coupled
chaotic maps predicts from the loop's own Lyapunov exponent.”

Evidence: All 18 Table VII brackets and five nonmonotonicity flags were
independently reduced from the complete distance arrays. They are finite
horizon/tolerance/start-dependent results, and some differ substantially
from c*. For example, the refined near-start 10^-8 criterion gives
(0.12,0.15] in cell 3 versus c*=0.089, and no pass through 0.19 in cell 2
versus c*=0.177. The transverse growth-rate comparison does reproduce,
with maximum discrepancy 0.0506046891. Sec. VIII and consult 24 properly
distinguish these claims. See `raw-evidence.json`, `synchronization`,
and [citations-review.md](citations-review.md), F2.

Correct value / wording: “Coupled copies have transverse growth rates
that approximately follow the prediction from the single-loop Lyapunov
exponent; observed synchronization thresholds additionally depend on
initialization, duration, tolerance and attractor selection.”

### F5. One cascade deviation reduction is 9.07, outside the stated range [must-fix]

Where: Sec. VI, Table VI discussion, lines 1102–1103.

Claim: after the first, the deviations “shrink by factors of 3.5 to 6.5
per flip.”

Evidence: `cascade_audit.py` reduces the nine saved flips and all seven
ratios. The successive reductions after the first ratio are approximately
**9.0653, 3.5188, 6.5386, 3.9264 and 5.2591**. The first of these is
omitted by the stated range. All seven tabulated ratios, the 0.008%
comparison, and the disclosed last-location sensitivity reproduce.
See [numerical-review.md](numerical-review.md) and `cascade-evidence.json`.

Correct value / wording: “After the first ratio the deviations have one
sign. From n=4 onward their reduction factors are approximately 3.52,
6.54, 3.93 and 5.26; the last reduction is not resolved by the stated
location uncertainty.” Alternatively report the wider approximately
3.5–9.1 descriptive range, with the same uncertainty caveat.

### F6. Nearby period labels do not establish fixed-start orbit identity [must-fix]

Where: Sec. VII, ditches, lines 1408–1414.

Claim: “on the patch and the line it is also the attractor that the map's
fixed start reaches.”

Evidence: `ditches_followup.py` verifies the saved continued-orbit gates
and compares the patch/lines with the nearest HR10 fixed-start scan cells.
593/595 solved patch neighbors and all 26+21 line neighbors are labeled
period six; two patch neighbors are unresolved. These are **nearby
parameters**, not same-parameter full-state orbit comparisons. Matching
period and six-token text cannot exclude a distinct coexisting orbit.
The two unresolved neighbors are not counterexamples at the continued
parameters. See `ditches-followup-evidence.json` and
[numerical-review.md](numerical-review.md).

Correct value / wording: “Across the sampled patch and lines the
continued orbit is attracting and retains its six-token text; nearby
fixed-start scans predominantly find period six.” State that full-state
identity with the fixed-start attractor was not checked at every continued
point. Do not extend attraction to every solved cell of the larger body
grid, which also includes unstable continued orbits.

### F7. Distinguish estimated location errors from bounds; round bounds outward [must-fix]

Where: Sec. V, fold precision, lines 850–866, and window-law limits,
lines 1006–1010.

Claim: edges are located “to within 2×10^-4 of their distances from the
fold”; critical multiplier within 3×10^-13; independent b agreement
to 2.2×10^-6 relatively.

Evidence: The P16–20 locations are extrapolations; the P22–32 boundaries
come from finite-grid period labels. The quoted grid uncertainty uses
spacing/sqrt(12), a uniform-within-cell standard-deviation model, not an
enclosing bound. Separately, the saved critical error is
3.006483950684924×10^-13, and the independent b relative discrepancy is
2.2068638969017986×10^-6. See [fold-review.md](fold-review.md), S2–S3,
and its exact original/closed-receipt hashes.

Correct value / wording: “Estimated localization scales are about
2×10^-5–2×10^-4 of the distance from the fold, using extrapolated P16–20
locations and a uniform-within-cell error model for P22–32 grid
boundaries.” Use “within 3.1×10^-13” and “within 2.3×10^-6 relatively
in b,” or explicitly label the latter approximately 2.2×10^-6.
The grouping is must-fix because the first statement claims an unsupported
error bound; the small rounding corrections alone are minor.

### F8. Availability statements must match the submission state [must-fix]

Where: Data Availability, lines 1770–1778; Supplementary Material link,
lines 1746 onward.

Claim: materials “are openly available” in the repository and complete
certificate evidence “is archived at Zenodo.”

Evidence: The Task40 brief calls DOI 10.5281/zenodo.23488416 reserved and
unpublished. Applicable repository policy keeps the repository private.
Public URL fetch failures here do not establish a 404; the problem is
that the claimed public state has not been established. The deferred
package-content audit cannot itself establish publication.

Correct value / wording: Before submission, describe the actual authorized
state. If still private/unpublished, say the code, full report and evidence
are available from the author on request and that a DOI is reserved for a
planned deposit. If the author independently authorizes and completes
publication, verify the public landing pages before retaining the
present-tense claims. Zenodo fixes may follow the article review; no
publication or visibility change is authorized by this report.

### F9. “Most” far-start period-two outcomes lacks a verified denominator [should-fix]

Where: Sec. VIII, lines 1615–1620.

Claim: “from far starts most first-cell pairs synchronized onto a
coexisting period-2 orbit.”

Evidence: Selected saved full-state diagnostics verify two synchronized
period-two examples, at c=0.12 and 0.30. A synchronized c=0.24 example has
no return below tolerance. The records do not contain full-state
classification for every member of the claimed subset. Token patterns
alone do not establish primitive full-state period two. See
[raw-review.md](raw-review.md), A7.

Correct value / wording: Change “most” to “some” unless the complete
denominator and full-state classifications are supplied. Preserve the
separate observed joint-periodic examples.

### F10. A bifurcation need not appear from every start [should-fix]

Where: Sec. VII, kite edge, lines 1463–1465.

Claim: a bifurcation “would appear for any start.”

Evidence: In a multistable system a fixed-start diagram can miss an
invariant-set bifurcation because that start reaches another attractor.
This logical implication is not supplied by the checked local coexistence
or saddle records. The latter independently support the basin-edge account.

Correct value / wording: “Bifurcation curves are properties of the map,
whereas a fixed-start scan shows only attractors reached from that start.
Together with the coexisting-orbit and saddle checks, the disappearance
of this edge supports a basin-boundary interpretation.”

### F11. Name the tracked block and keep the trace result local [should-fix]

Where: Sec. VII, lines 1415–1438; Table VIII, pseudo-superstable line row,
around line 1675.

Claim: “the two leading directions” and “the ditches are the zero curves
of the trace of the leading two-dimensional block.”

Evidence: Independent SVD/projection-overlap tracking gives 852 cells,
38 overlaps below 0.9, 849 sign agreements, and 42/42 and 60/64 zero-set
matches. Selected Ritz pairs are (0,1) in 723 cells, (0,2) in 63,
and (1,2) in 66; they are not always the two locally largest-modulus
directions. The trace saddle is
(0.07318804087584509, −0.15208331868800173), value 0.044884483163678275.
The leading-real-multiplier saddle is
(0.07319482971306128, −0.1521384898381376), value 0.06736198957428159.
No determinant saddle was found under the stated fit/filter test.
See `delta_ditch_audit.py`, `delta-evidence.json`, and
[numerical-review.md](numerical-review.md).

Correct value / wording: “A two-dimensional block initialized from
leading Ritz directions and tracked by subspace overlap.” In Table VIII
say “the sampled patch numerically supports a trace-zero/saddle account.”
Retain the properly disclosed distinction between the failed pre-data
determinant-saddle test and the retrospective trace analysis. This is not
a global absence theorem for determinant saddles.

### F12. State the assumptions behind the cited mixture threshold [should-fix]

Where: Sec. II, steering and temperature, lines 291–293.

Claim: the product's critical value 2 separates mixture washout from
one-token takeover.

Evidence: Backour's primary v1 requires the unbiased two-token reduction
with linear logit transport and positive constant coupling for this
balanced-state threshold. Variable coupling needs further uniform
conditions. See [citations-review.md](citations-review.md), F3.

Correct value / wording: “In Backour's unbiased two-token reduction with
linear logit transport and positive constant coupling, inverse
temperature rescales the coupling; crossing 2 destabilizes the balanced
mixture.”

### F13. Separate what descent proves from a nonconvergence assertion [should-fix]

Where: Sec. III, “Without the transformer, no chaos,” lines 445–448.

Claim: where fixed points are not isolated, “an orbit need not converge
to a single one.”

Evidence: The displayed descent estimate and boundedness prove vanishing
steps and fixed accumulation points. They alone do not supply an example
of nonconvergence for this particular analytic log-sum-exp map. The
accepted-review caution about the limits of a descent argument should
not become a stronger model-specific nonconvergence assertion.

Correct value / wording: “This descent estimate alone does not establish
convergence to a single fixed point when fixed points are nonisolated.”
This review does not claim to settle a stronger convergence theorem.

### F14. Broaden the historical coupling attribution unless an exact passage is supplied [should-fix]

Where: Sec. VIII, transverse stability, lines 1554–1556.

Claim: attribution of this precise discrete output-coupling condition
to Fujisaka and Yamada.

Evidence: The primary publisher record verifies the article and its
Lyapunov-stability framework, but accessible primary text did not verify
this exact coupling convention. Consult 24 requested an exact passage
or broader attribution. The manuscript derives its own formula correctly;
this is an unverified attribution, not a demonstrated false reference.

Correct value / wording: “The calculation above gives the
transverse-stability condition for our symmetric output coupling; it
follows the Lyapunov-stability framework of Fujisaka and Yamada.”
See [citations-review.md](citations-review.md), F5.

### F15. Two small editorial/layout fixes remain [should-fix]

Where: Sec. VII, lines 1382–1383, PDF page 24; Fig. 22, first body mention
page 22 and display page 24.

Claim: “Near four of these five come in two pairs”; requested figure
placement within one page unless there is a reason.

Evidence: All 34 rendered pages were inspected. The stray “Near” is
literal source/PDF text. Fig. 22 is the sole figure more than one page
from its first body mention; the intervening page contains the large
canonical anatomy Fig. 21. See [bundle-review.md](bundle-review.md).

Correct value / wording: “Four of these five come in two pairs.”
Bring Fig. 22 forward if feasible, or retain its delay with that explicit
layout reason. Recheck the final revised bundle's pagination.

### F16. Preserve audit limits and the two failed audit attempts [note]

Where: Task40 evidence records; relevant article precision/provenance claims.

Claim/check: What this independent audit establishes.

Evidence: Saved-data reduction is independent arithmetic/tracking, not
independent model, derivative or hardware-execution reconstruction.
The initial `ditches-001` exact-axis check failed; a separately disclosed
follow-up records the one-ULP differences. The `bundle-001` outer Snap
launcher refused before Python; a separately authorized host invocation
completed. All failures and subsequent source/launch amendments are
retained. P384/P768 action-based Ritz residuals actually exist
(1.2513×10^-15 and 4.4434×10^-15); the external-review log's “not computed”
statement is stale. Original synchronization metadata declares dirty
producer tools, whose exact dirty bytes are not authenticated here.

Correct value / wording: Do not erase these limits or report the stale
Ritz omission as a current manuscript defect. Do not call the CPU/GPU
comparison bit-identical-input verification. No new numerical certificate,
global chaos proof or all-start attractor claim follows from this audit.

### F17. Single-sample passages become rare rather than vanish [must-fix]

Where: Discussion, “Beyond the one-token loop,” lines 1720–1723.

Claim: phases shrink “from 120 steps to 7.7 and 10.6 steps” and “vanish
for single samples.”

Evidence: `discussion_delta_audit.py` independently reduces all four
saved `bridge-*/walk.npz` archives under the original probability-mask
and run-length-at-least-six rule. At the three offsets, single sampling
retains **1, 1 and 3 phases**, with lengths 7, 8, and 6/6/6 steps.
The 64-sample means are 10.5193, 10.3237 and 10.7224; the eight-sample
means are 7.8145, 7.7101 and 7.5968. The deterministic 120-step baseline
belongs to the smallest offset; the second offset gives 63 steps and
the third gives no retained phase. These are original float32,
probability-defined observations, not the geometric normal-form channel
in F1. Evidence: `discussion-delta-evidence.json`.

Correct value / wording: “At the smallest tested offset, the mean
probability-defined laminar phase shrinks from 120 steps with deterministic
feedback to 7.8 and 10.5 steps with eight and 64 samples. Across the three
offsets the sampled means are approximately 7.6–7.8 and 10.3–10.7 steps.
With single samples such phases become rare: only one to three are
recorded per offset.” Avoid an exact disappearance claim. The original
statistic includes left-censored runs and excludes right-censored ones;
retain that convention or explicitly disclose a revised one.

## Completed, partial and deferred checks

| Requested check | Status and evidence |
|---|---|
| A1 answer-test and reset correction | Completed saved answer/count reduction and comparator source/test inspection: all quoted rates reproduce, including reset changes. Old model runs and comparator tests were not rerun. Budget-hit classification uses the recorded answer-length convention, not an independently observed termination cause. `raw-review.md`, A1. |
| A2 original ditches and lines A/B | Completed saved row gates, spectra ranges, sign/complex-pair observations and quoted 595/26/21 counts. Larger body includes unsolved/unstable rows, retained. Full model/monodromy not recomputed. `numerical-review.md`. |
| A3 fine GPU/local comparison | Completed all 225 matched cells; all solved, real positive leading multipliers, text identities and quoted ranges. Maximum next-modulus difference 3.764×10^-13. Exact alpha axes match; two temperature rows differ by 2.7755575615628914×10^-17. Failed strict comparison preserved. |
| A4 decisive DELTA2D test | Completed independent SVD and projection-overlap tracking, zero-set counts, and quadratic stationary-point analysis from saved three Ritz vectors. Same declared seed/traversal/objective disclosed. No producer imports; no fresh original Ritz-vector computation. Low-overlap exceptions retained. Pre-/post-data distinction checked. |
| A5 A/B pseudo-crossings | Completed all 326 saved curve rows and four quoted neighborhood gates; 24 other unconverged-inner-solver rows retained. Bounds 0.0022/0.0042 pass. Nearby spectra are not exact common-parameter crossing certification. |
| A6 cusp unfolding | Completed six records, twelve bracket endpoints, determinant/singular-value/scaled-coordinate reductions and final tolerances. K coordinates are (log10 T, 10 alpha). Final Newton history is absent from compact locator outputs and remains unverified. F2 applies. |
| A7 synchronization | Completed all 18 brackets, five stars, three exponents/c*, 21 transverse comparisons, burst counts and saved selected joint-return diagnostics. Full-state primitive-period reconstruction and the “most” denominator remain partial; dirty producer source not authenticated. |
| A8 hook tip / kite edge | Completed saved saddle, local coexistence, all 34 distance columns, exit/closest-distance fits and hook-tip interpolation/extrapolations. Exit slope convention uses last exit from relative radius 10^-3. Random-start 128×128 absence of the edge remains a finite-grid visual claim, not freshly re-reduced. |
| A9 Zenodo | **Deferred until after this article report is delivered, as the human requested.** No package-content pass is implied. A later addendum will record checksum/closure/text screening results. |
| B1 Sec. V | Completed fresh fold/passage/window reductions plus exact reuse of closed independent audits. All 12 Table III observed/predicted pairs, all 48 Table V measurements and Table IV reductions match under the stated definitions. Failed prospective criteria retained. Free-exponent nonlinear fits 2.13–2.45 and 1–3% fit misfits were not independently refitted and remain partial. |
| B1 Sec. VI | Completed nine flip records, seven ratios, quoted sensitivities and alternate-cut comparisons, FT scan dimensions/horizons and rejected linear-fit attempt. No independent monodromy/JVP evaluation. F5 applies; finite-order consistency is not universality proof. |
| B1 Sec. VII–VIII | Covered by A2–A8 above, including per-row gates and evidence limits. Old previously audited anatomy/model computations were not repeated. |
| B1 Discussion, Table VIII, Conclusion | Claims calibration completed, with F10–F13/F17 and unchanged Task39 evidence reused. Three saved 600-step, four-direction spectra give exactly one positive exponent each and KY dimensions 1.13311/1.42779/1.20843; the negative range is −2.00307 to −0.821936, within the article's outward enclosure. All four sampled-bridge archives freshly reduced. No blanket “every number independently recomputed” claim is made. |
| B1 Data Availability | Completed wording/state check; F8 applies. Package contents deferred; no publication action. |
| B2 three external reviews | Complete check of the two available accepted-point summaries and full consult 24 plus adjudication. Original 27-/18-page review documents were not present, so full original-review coverage cannot be claimed. See `citations-review.md` accepted-point ledger. Chaos/flip/Feigenbaum caveats largely applied; abstract/sync summary and descent wording need fixes. |
| B3 figures/tables/references | All 34 pages visually inspected; 27 figures/eight tables, no missing/duplicate labels or final unresolved references. Fig. 1 on page 1. F15 records layout exceptions. Visual checking does not independently regenerate every unchanged figure's underlying experiment. |
| B3 arXiv bundle | PASS for the frozen bundle: all 33 members match the frozen source/graphics or declared comment normalization; regenerated bibliography matches. Canonical and bundle builds give exactly matching text and all 34 page renders. Complete PDF bytes match after only two date-field and one trailer-ID substitutions. New author edits require a fresh final bundle/build check. |
| B4 added citations | All six newly added keys checked against primary publisher/arXiv records, with claim-specific checks. Changed existing metadata checked separately: Yuan's new page range and Mira's added series volume remain primary-unverified after access failures; no mismatch inferred. Unchanged bibliography not rerun. |
| Old completed tasks / paid compute | No old controls, certificates, model experiments or watcher restarted. No paid computation, GPU job, external review call, publication, or other-owner process action. |

## Evidence and execution

Component reports contain exact input inventories, hashes, conventions and
limits: [raw-review.md](raw-review.md),
[fold-review.md](fold-review.md),
[numerical-review.md](numerical-review.md),
[discussion-delta-review.md](discussion-delta-review.md),
[citations-review.md](citations-review.md),
[bundle-review.md](bundle-review.md).
Scripts, audit outputs and named parent receipts are in this same
Task40 directory. Original raw evidence remains at the paths in their
input manifests. Large PDF/build scratch remains under
`/data2/neural-shrimp-data/scratch/audit-codex-1020/`.

Executable audit sources were privately checkpointed before execution.
Completed bounded workers (`raw-001`, `delta-001`, `cascade-001`,
`ditches-002`, `fold-001`, `bundle-002`, `discussion-001`) exited 0 without timeout and were
reaped by their parents. The unsuccessful `ditches-001` and pre-Python
bundle launch refusal remain intact. CPU-only saved-data work used one
numerical thread per worker; the original GPU runs were merely read.
These execution facts do not authenticate unsaved historical source bytes
or establish new scientific admission.

The article is suitable to proceed **after** the must-fix wording/numerical
corrections, with the partial coverage above disclosed. This review does
not edit or submit it and does not transfer approval to an unseen revised
snapshot.
