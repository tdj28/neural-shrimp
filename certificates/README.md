# Rigorous orbit certificates

This C1 pilot asks whether full-state neural-network training cycles already
observed numerically exist and attract nearby states in the exact mathematical
map. It does not test a training benefit or shrimp geometry. The pilot is
exploratory. The independent reviews and their different scopes are recorded below.

The checker uses Arb real balls through python-flint. Every stored binary64
constant (dataset, scale, ridge and learning rate) is interpreted as its exact
dyadic rational. The loss is `loss_scale * mean((prediction-y)^2) +
ridge/2 * sum(theta^2)` and the real map is `theta - P gradient(loss)`, with
`tanh` in real arithmetic. This is not a theorem about finite-precision training.
The input exporter checks regenerated dataset hashes and network values against
the unchanged EXP-028 plan, then stores the dataset itself for offline checking.

For `H=F^p-I`, the checker refines a dyadic proposal, encloses the map and exact
analytic Jacobian on a box `X`, and takes an exact dyadic approximate inverse
`R`. It requires both strict Krawczyk inclusion
`x-R H(x)+(I-R H'(X))(X-x) < interior(X)` and infinity norm
`||I-R H'(X)|| < 1`. These establish that `T(z)=z-R H(z)` maps `X` strictly
inside itself and is a contraction. The derivative bound also makes `R`
nonsingular, so the unique fixed point is a zero of `H`. Proper divisors of
`p` are excluded by a coordinate separating `F^d(X)` from `X`. Attraction
requires a rigorous bound `||M^k||inf < 1` for an interval enclosure of the
monodromy at the proved periodic point. Thus every actual monodromy in the
enclosure has spectral radius below one. Float eigenvalues never decide a gate.

Direct interval iteration can fail through wrapping. A failed enclosure is
inconclusive and is retained. Multiple shooting is a possible later amendment,
not part of this initial checker. Passing artifacts carry evidence rung 3 for
the narrow mathematical statement; the research status remains an exploratory
certificate pilot with independent review complete. The checker and upstream Arb library
are not formalized in a proof assistant.

## Execution and replay

Use the project environment (python-flint 0.9.0 / FLINT 3.6.0 at development).
Code must be committed and a research-log run card recorded before target runs.
`tools/certify_c1.py prepare` exports inputs from an existing EXP-028 numerical
receipt; `run` writes an exclusive new attempt under the external data store.
The first selected candidate was row 121, S2-28011-zoom component 936, period
10: the best-conditioned embedded period-5/10 candidate by existing numerical
`condition(M-I)`. Scope review found it was absent from the final historical
window qualifying lists. Its orbit proof is retained with no surrounding-chaos
claim. The amended target was row 35, S2-28011 component 140, period 10: the
best-conditioned period-5/10 candidate in the intersection of the unchanged
EXP-028 summary and independent audit qualifying lists. Both selections and
the amendment are disclosed in the research log; neither is confirmatory.

The compact receipts bind inputs, all raw files, source commit and source
hashes. Then one offline command regenerates all enclosures and compares them
with the saved result, refusing missing/corrupt inputs, modified checker code,
changed verdicts or any failed proof gate:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 \
  /data2/neural-shrimp/.venv/bin/python tools/verify_certificates.py \
  certificates/c1-row35-receipt.json certificates/c1-row121-receipt.json \
  certificates/s1r-period2-receipt.json certificates/s2-period18-receipt.json
```

The receipt's external root locates raw artifacts on zeta. A Git push backs up
source and the receipt; it does not independently back up the external raw files.

| Component | Evidence | Artifact / assumption |
|---|---|---|
| Map/Jacobian implementation | Test cross-check, not a theorem by itself | `tests/test_rigorous_orbits.py`; independent frozen NumPy implementation and enclosed interior samples |
| Existence/local uniqueness | Rung 3 gates pass; exploratory, independently reviewed | The four original C1 receipts and the separate period36 wrapper below; Banach contraction, analytic map implementation and Arb/FLINT trusted |
| Primitive period | Rung 3 gates pass; exploratory, independently reviewed | The period-10 receipts exclude divisors 1, 2 and 5; period18 excludes 1, 2, 3, 6 and 9; S1R excludes divisor 1, by disjoint coordinates |
| Attraction | Rung 3 gates pass; exploratory, independently reviewed | Period-10 monodromy powers 64, period18 power16 and S1R power 2048 have infinity norms below one |
| Chaos, window geometry, invariant circles, utility | Not established by this track | Existing evidence retains its own scope and labels |

## Pilot outcome (7 October 2026, user time zone)

**Behavioral TLDR:** for two selected learning-rate settings of a 13-parameter tanh network
(the same S2 dataset and architecture), the exact-real training map
has an attracting full-state period-10 orbit. Thus small perturbations near
those mathematical cycles decay while the full parameter state keeps cycling.
One orbit is in a historically qualifying EXP-028 window. This result does
not yet establish a shrimp-specific behavioral effect, a useful readout,
surrounding chaos rigorously, or robustness to finite-precision arithmetic.
The computation passes proof gates and independent mathematical/code review; target selection
remains exploratory. The review record is `c1-independent-review.json`.

| Attempt | Target and purpose | Outcome | Compact receipt |
|---|---|---|---|
| 001 | Row 121, sequential matrix powers | Inclusion and primitive-period gates pass; attraction inconclusive through wrapping | `history/c1-attempt-001-receipt.json` |
| 002 | Row 121, disclosed repeated-squaring amendment | All gates pass; superseded only for checker source binding | `history/c1-attempt-002-receipt.json` |
| 003 | Row 35, qualifying-window target | All gates pass; authoritative for this orbit | `c1-row35-receipt.json` |
| 004 | Row 121 replay after receipt hardening | All gates pass; authoritative for this orbit; not an additional orbit | `c1-row121-receipt.json` |

At 256-bit midpoint precision with box coordinate radius `2^-80`, the
Krawczyk contraction bounds are below `9e-13` (row 35) and `8.1e-14`
(row 121). The rigorous upper bounds for `||M^64||inf` are below `0.056`
and `0.108`, respectively. These are upward-rounded summaries of the exact
dyadic bounds in the receipts, regenerated by the offline command above.
Attempt 004 exactly reproduces attempt 002's mathematical result, including
every enclosure; only provenance/runtime changes. Each actual pilot used one
CPU thread and less than one second; no paid compute was used.

The initial failed attraction enclosure is an implementation limitation, not
evidence of instability. Its raw output and exact original code commit remain
available. Historical receipts must be replayed with their bound source commit;
the offline command above checks the four authoritative current certificates.
All raw attempts are retained under the external root in each receipt, and the
code prevents reusing an output directory. The tests include independent
NumPy map/Jacobian comparisons, interior enclosure samples, exact linear
positive controls, false primitive-period and unstable controls, and corrupt
input/source-inventory/summary rejection. None replaces independent review.

## Independent review (8 October 2026 UTC)

The source-bound review in `c1-independent-review.json` reports no blocking
findings for producer tip `32528cbbb4a3313bf82f63a9a812584ad754a82e`. Both S2
authoritative receipts replayed; input bindings, stored Krawczyk/divisor/power
witnesses and an independent second-order Arb differentiation of the objective
were checked. Both proofs also passed at 384 bits in smaller nested boxes.
This shares the Arb library and is not proof-assistant formalization. Immutable
receipts retain their original creation-time review-pending labels; this
companion record supplies the subsequent review status.


## Full 72D S1R certificate (8 October 2026 UTC)

`s1r-period2-receipt.json` proves local existence, uniqueness, primitive period
two and attraction in the 72-parameter S1R-28002 map in a certified box near selected saved node 0004.
The exact learning rates are `1870594557782953/140737488355328` (hidden) and
`2385682499131913/562949953421312` (output); the input file binds all dataset
values, scales and the exact dyadic interpretation of ridge 0.001.
The fixed settings were 256-bit precision, radius `2^-80`, five proposal Newton
steps and maximum monodromy power 4096. The successful witness uses power 2048
with an exact upper bound below 0.556. The unchanged C1 kernel took 3.30s on one
CPU thread; no paid compute or GPU was used. Selection remains exploratory.

The companion `../docs/reviews/s1r72-certificate-outcome-receipt.json` binds the
original selected arrays and exact-real inputs, checks the source/resource
records, replays the proof once and independently computes an exact-Fraction
whole-box proximity bound. The declared normalized threshold 1e-7 passes. This
does not assert that the original floating-point state lies inside the proof
box, prove historical branch ancestry, enclose a parameter interval, or prove
the bifurcation/window geometry. Proximity is separate from proof validity.

This review reuses the already reviewed Arb checker and theorem arguments.
The separate second-order Arb differentiation and 384-bit nested-box review
described above apply to the two period-10 S2 proofs, not this 72D outcome audit.
Reproduce the outcome auditor from its recorded historical checkout: later
observer additions to `gdmaps.py` changed auxiliary source hashes without
changing the three C1 proof-kernel files. The ordinary offline proof command
above still uses those unchanged kernel files. Original receipts retain their
creation-time pending labels; the companion review supplies subsequent status.

## Period18 in the selected anatomy candidate (9 October 2026 UTC)

Before the later ray-51 addition below, the four independently reviewed individual orbit proofs comprised two
period-10/13D, one period-18/13D and one period-2/72D orbit. The new canonical
receipt is `s2-period18-receipt.json`, selected from S2-28011-zoom component587,
EXP-028 row111. It proves local existence/uniqueness, primitive period18 and
full-state attraction at the supplied exact dyadic data and fixed layer rates.
The unchanged C1 checker encloses `||M^16||inf < 0.726`. Historical ray/map anatomy
remains numerical; the orbit proof does not certify complete shrimp geometry.

Raw evidence lives at
`/data2/neural-shrimp-data/explore/certificates/s2-period18-2026-10-09/attempt-001`.
The producer companion is `../docs/reviews/s2-period18-producer-receipt.json`;
the completed review is `../docs/reviews/s2-period18-independent-review.json`
with raw review `/data2/neural-shrimp-data/audits/shrimp-period18/attempt-001`.
It replays the unchanged checker, checks stored witnesses and exact whole-box
proximity, compares two samples with independent objective differentiation,
and passes a 384-bit proof in a smaller nested box. These checks share Arb/FLINT;
no proof-assistant formalization, binary64 orbit theorem, parameter continuation,
boundary theorem, surrounding-chaos theorem or utility is implied. The original
receipt retains its creation-time pending label. The command above now includes
the new receipt without changing the proof-kernel source files.


## Local primitive period-18 fold (9 October 2026 UTC)

`s2-period18-fold-receipt.json` proves a local nondegenerate saddle-node of
primitive period-18 cycles in the same exact-real 13D S2 map, with fixed output
rate `1848862009810323/2^51` and actual hidden rate in the outward interval
`[75.76429203111452,75.76429203111453]`. The normalized 27D system is
`(R, R_theta v, (v^T v-1)/2)=0`, where `R=F_eta^18-I`. Interval inclusion,
proper-divisor exclusion, a uniformly invertible transpose border, nonzero
`a_eta=w^T R_eta` and `c=w^T R_theta_theta[v,v]` establish the local fold
hypotheses. With `v^T v=w^T v=1`, the scalar quadratic coefficient is `c/2`.
The deflated monodromy `Q=M-vw^T` satisfies `||Q^32||inf<0.036`, so the other
12 multipliers lie strictly inside the unit circle. The fold has a neutral +1
multiplier and is **not an additional attracting-orbit certificate**.

The radius-`2^-80` first attempt passed all gates except the complementary
power bound. It remains in `../docs/reviews/s2-period18-fold-attempt001-receipt.json`.
The disclosed same-target radius-`2^-100` amendment passed at unchanged 256-bit
precision. The producer bundle is
`/data2/neural-shrimp-data/explore/certificates/s2-period18-fold-2026-10-09/attempt-002-target`.
The independent companion is
`../docs/reviews/s2-period18-fold-certificate-independent-review.json`, with
review bundle `/data2/neural-shrimp-data/audits/s2-period18-rigorous-fold/attempt-001`.
It checks exact replay and saved interval arithmetic and passes a 384-bit proof
in a strictly nested box. These checks share the mixed-jet kernel and Arb/FLINT;
they are neither separate automatic differentiation nor a proof-assistant proof.
No finite-detuning enclosure, connection to the earlier interior orbit, complete
window boundary, resonance ancestry or utility is established.

The manuscript's `scripts/verify.py` now includes this replay, preserves an
exclusive scratch directory on success or failure, and prints its location.
To replay it separately from a clean checkout with unchanged bound source:

```sh
fold_replay_root=$(mktemp -d /tmp/neural-shrimp-fold-replay.XXXXXX)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=5s 135s nice -n 10 \
  /data2/neural-shrimp/.venv/bin/python tools/certify_s2_period18_fold.py \
  --execute --mode replay \
  --bundle /data2/neural-shrimp-data/explore/certificates/s2-period18-fold-2026-10-09/attempt-002-target/run \
  --manifest-sha256 9677cd6817d3e3cdfb45dcc10656a5660b79fec2b98aa7365ab268c705c36e5b \
  --out "$fold_replay_root/run" >"$fold_replay_root/process.log" 2>&1
fold_replay_exit=$?
printf '%s\n' "$fold_replay_exit" >"$fold_replay_root/exit-code.txt"
printf 'Retained replay: %s (exit %s)\n' "$fold_replay_root" "$fold_replay_exit"
```

The original producer receipt retains its creation-time review-pending label;
the independent companion supplies the completed review status.


## Local primitive period-18 subcritical flip (9 October 2026 UTC)

`s2-period18-flip-receipt.json` proves a local nondegenerate flip of primitive
period-18 cycles on the exact historical ray `eta_i(s)=exp(log(10)*(ell_i+d_i*s))`,
with both layer rates varying. The receipt records all four exact dyadic ray
constants. The direction constants are the rounded binary64 scale-direction
products, then interpreted exactly. The resulting exponential rates, including
at s=0, are not asserted equal to the earlier dyadic center-certificate rates.

The 27-dimensional augmented system encloses `R=0`, `(M+I)q=0`, and unit q.
A validated left border gives an algebraically simple -1 multiplier and
`p^T q=1`. A validated inverse of `A=M-I` gives the moving branch derivative.
The total multiplier derivative is strictly negative, approximately -51.234629;
the cubic coefficient `b=p^T C(q,q,q)/6-p^T B(q,A^-1 B(q,q))/2` is strictly
negative, approximately -78.084775. Directional third-order AD differentiates
the explicit gradient update; no finite-difference derivative enters the proof.
The deflated monodromy `Q=M+qp^T` satisfies `||Q^64||inf < 0.046042`.
All proper divisors 1, 2, 3, 6 and 9 are excluded.

Thus the flip is subcritical: sufficiently nearby primitive36 cycles on the
lower-s side are unstable, while the parent18 branch is locally attracting
there. The flip itself is neutral, not an additional attracting-orbit certificate.
This is not a finite-detuning36 enclosure, proof of first loss of attraction,
validated center connection or whole-window geometry. The numerical precursor
retains an earlier complex instability and re-entry on the same carried ray.

The single target at 256 bits, radius2^-100 and five Newton proposals passed
all eight gates. Raw evidence is retained at
`/data2/neural-shrimp-data/explore/certificates/s2-period18-flip-2026-10-09/attempt-001-target`.
The producer receipt retains its creation-time pending independent-review status;
the passed companion `../docs/reviews/s2-period18-flip-independent-review.json` records byte-identical replay, independent saved-ball reductions, and a strictly nested 384-bit/radius2^-110/five-Newton proof. Its first pre-proof schema failure is preserved. The reviewer authored the earlier fold primitive and numerical proposal, but not the new flip kernel; shared analytic differentiation and Arb remain trusted. The proof trusts the analytic gradient,
directional-jet implementation, the smooth-map theorem and Arb/FLINT.

Offline replay, retaining both success and failure output:

```bash
flip_replay_root=$(mktemp -d /tmp/neural-shrimp-flip-replay.XXXXXX)
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=5s 135s nice -n 10 \
  /data2/neural-shrimp/.venv/bin/python tools/certify_s2_period18_flip.py \
  --execute --mode replay \
  --bundle /data2/neural-shrimp-data/explore/certificates/s2-period18-flip-2026-10-09/attempt-001-target/run \
  --manifest-sha256 7fdd17d546a3514b49fdb21178fad6caf29ab93d7061af883ea55428d5df3e59 \
  --out "$flip_replay_root/run" >"$flip_replay_root/process.log" 2>&1
flip_replay_exit=$?
printf '%s\n' "$flip_replay_exit" >"$flip_replay_root/exit-code.txt"
printf 'Retained replay: %s (exit %s)\n' "$flip_replay_root" "$flip_replay_exit"
```

## Ray-51 supercritical flip and separate attracting period36 (9 October 2026 UTC)

The two new canonical receipts are `s2-ray51-flip-receipt.json` and
`s2-period36-receipt.json`. The exact exponential-family local flip has alpha<0,
b>0 and an attracting complement (`||Q^16||inf < 0.429488`); it gives a local
attracting primitive36 branch on the increasing-s side, with no finite detuning
specified. A separate fixed-dyadic map has an attracting primitive36 orbit with
`||M^512||inf < 0.117942`; all proper divisors 1,2,3,4,6,9,12,18 are excluded.
This raises the individually certified attracting-orbit count to five. The
exact family and rounded-rate map are distinct definitions; their connection
has not been validated. Neither neutral flip increases that count.

`../docs/reviews/s2-ray51-certificates-independent-review.json` records exact
input reconstruction, byte-identical original replay, independent saved-ball
arithmetic and strictly nested fixed proofs (flip384bits/radius2^-110;
period36 512bits/radius2^-170, five Newton proposals each). The reviewer authored
the shared flip kernel; independent adapter review and separately reviewed
kernel mathematics are distinguished from shared AD/Arb trust. Original
pending-at-creation labels are preserved in the producer receipts.

The new receipts have a wrapper schema distinct from the original C1 receipts.
Replay each with `tools/certify_s2_ray51.py --kind flip|period36 --mode replay`,
the corresponding receipt's `target.root/run` and externally pinned
`target.run_manifest.sha256`, and a new exclusive `--out` directory. The
receipt records its concrete command. `paper/gd-route/scripts/verify.py` now
retains a separate persistent command/log/exit bundle for each replay; it does
not pass these wrapper receipts to the older C1 receipt reader.
