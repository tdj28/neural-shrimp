# Task18 independent loop-gain run001 (prepared)

Status: source preparation only. No target/model run has been launched. The
prepared plan must be reviewed and privately checkpointed before one execution.

The human requests Tasks18 and19 (including the cross-cell addendum), followed
by17,16,15. Task18 is an exploratory independent numerical check after exposure
to the producer summary and analysis source at93dfb7617df34e2878da4ee0c6b9ded0b0e54362.
No producer computation is imported. This is not an outcome-blind confirmation.

## Fixed computation

Use unchanged own ManualQwen source, exact stored model/vector hash pins and
37 prefix IDs. Philosophy steering after zero-based layer12 also acts on the
prefix; alpha0.071674 and log10(T)=-0.115; full24 layers/896 state/151936 vocabulary.
Generate1000 transient steps from the fixed embedding start, then save x0..x8.
Require relative closure<=1e-10 and every pair of the8 phases separated by>1e-8
of their mean norm. These are numerical qualification gates, not certification.

For each phase retain final normalized h, full softmax p, image, all896 native
joint(h,F) basis returns in batches16, full J_h/DF and centered all-token Sigma_p.
Independent8 random original-ManualQwen.forward JVPs per phase (PCG64 seeds18001
through18008) check factorization and dense columns. Relative derivative gates
are1e-9, original-vs-new forward1e-12. The covariance symmetry/PSD checks are
rounding checks with the exact constants in analysis.py. All singular values,
all covariance eigenvalues, and every cyclic-start chronological monodromy,
signed/complex eigenvalues and singular values are retained. Phase0 chronological product is also compared with the unchanged original
ManualQwen.period_map(period=8) forward-AD on3 deterministic unit directions
(PCG64 seed18009), retaining primals and vector returns; relative derivative
gate1e-8 and primal gate1e-12. No leading-only Arnoldi approximation. Step-norm products and every second/first ratio are saved.

The final-normalized-hidden wrapper reuses the accepted primitive blocks;
original ManualQwen.forward is an additional independent code path at every
phase. This is an independent implementation from producer tools, not a wholly
independent software stack. Both use PyTorch arithmetic. Fixed prefix and
parameter derivatives are state-only. No coefficient, Newton, interval or model
training work occurs.

Own outputs are sealed before reading producer orbit-jacobians arrays. A later
finite comparison must align phase and report discrepancies: singular norms
can depend on the cyclic starting phase. It must retain the signed second
multiplier. No source/data/gate adjustment or silent retry follows disagreement.

## Finite resource basis

One CPU2 worker plus one supervisor reserves3 CPU slots; nice10,12GiB hard AS,
3600s external wall deadline plus5s TERM/5s KILL cleanup; maximum512MiB output
including16MiB worker closing reserve. Startup available RAM>=28GiB and runtime
floor16GiB; disk>=16GiB. CPU-only, no GPU/pod/paid action. Other auxiliary numerical
work is serialized until Root accepts a combined resource card.

The accepted Task14 period7 profile receipt75523fb2 and compact
codex/cusp-crosscheck/profile-001-receipt.json report dense896 time777.9554249s,
period7 map1.6170305s, peak5859160KiB. Dividing by7 is an extrapolation, not a
measurement of this new state: eight one-step dense matrices~889.092s and1000
forward steps~231.004s, total~1120.096s. Three composed JVPs add~25.81s from the measured
period7 JVP7.5279334s scaled by8/7; combined~1145.91s. Joint hidden output, full-vocabulary
centered covariance, eigensolvers/SVDs and cyclic products have not been timed;
the remaining2454.1s is an explicit overhead/uncertainty allowance, not a runtime
guarantee. No new timing run is required solely for these new matrix operations.
The retained per-phase arrays are estimated~47MiB, plus~50MiB cyclic matrices,
small spectra/directions/cache/metadata, below512MiB; the hard cap remains binding.
Prior peak5.588GiB is below12GiB but is not a guaranteed peak for the new run.

## Ownership and preservation

Raw target root: /data2/neural-shrimp-data/codex/llm-loop-gain-crosscheck/run-001.
Launch with the same admitted Python/backend as prior own runs. External
supervisor must snapshot every source/plan/card at its private Git commit,
establish limits before importing Torch, create fresh owned sessions and retain
PID birth/UID/PGID/SID, stdout/stderr, receipt and group closure. The worker's
cooperative checks do not interrupt a native call; outer timeout is authoritative.
Full actual capsule/supervisor source, launch command and preflight pins will be
added at admission. Missing/failed rows remain explicit. No automatic retry,
wall extension or changes to frozen earlier evidence. Status and certificate
are distinct: computational completion is not an interval certificate.
