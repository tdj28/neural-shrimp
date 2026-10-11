# CIFAR input-span diagnostic, 8 October 2026 UTC

**TLDR:** We may be spending the orbit solver on slow weight decay in directions
that do not affect the training predictions. This is an algebraic possibility,
not an observed explanation of the CIFAR failure. The next diagnostic uses only
already saved subset-A states and derivatives. It neither trains nor corrects a
model and makes no orbit, chaos, stability or utility claim.

## Exact-map identity to review

Write the first-layer matrix as W = Q B + N, where Q has orthonormal columns,
range(X^T) is contained in range(Q), and Q^T N = 0. For the current tanh MLP,
the data loss depends on W only through XW = XQ B. The isotropic ridge term
splits as lambda/2 (||B||_F^2 + ||N||_F^2). Thus in these orthogonal coordinates
plain full-batch GD is the direct product

    (z, N) -> (G(z), q N), q = 1 - eta lambda,

where z comprises B and all other parameters. This follows by differentiating
the split loss; neither linearizing tanh nor freezing the second layer is
required. The data-dependent gradient with respect to W is X^T times a matrix.
Consequently, for the period-p residual R_p = F^p - identity,

    (R_p)_perp = (q^p - 1) N,
    (D R_p v)_perp = (q^p - 1) v_perp.

If q^p != 1, every period-p point has N = 0. Its primitive period is preserved
by lifting a reduced orbit, and the full period-p derivative is the reduced
monodromy direct sum q^p I on the omitted coordinates. Attraction also requires
|q^p| < 1. A reduction does not discover an orbit, prove its existence or
qualify the remaining spectrum. With q^p = 1 the elimination is invalid;
with lambda = 0, invisible initial weights are constant and need not vanish.
The identity is a short analytic derivation, not a new ball certificate or
novelty claim. It does not apply without adjustment to adaptive optimizers,
changing input features, nonlinear parameter penalties or different metrics.
It preserves the training-input predictions; deleting N can change predictions
on unseen inputs, so it is not an authorized inference/training optimization.

For current d=3072, n=512, h=32, k=10, retaining all 512 thin-QR columns would
leave 512*32 + 32 + 32*10 + 10 = 16,746 active coordinates and 81,920 omitted
coordinates, compared with 98,666 originally. Do not assert exact rank511 from
standardization: stored binary64 centering need not have exactly zero mean.
Keeping all512 columns avoids a chosen singular-value truncation. In exact
arithmetic an orthonormal completion handles rank deficiency; the numerical
basis still needs its own residual checks.

Related background: Advani, Saxe and Sompolinsky (2020),
https://pmc.ncbi.nlm.nih.gov/articles/PMC7685244/, discusses frozen subspaces in
limited-data learning, including fixed-first-layer nonlinear networks. This
is background, not a citation for our particular period-p reduction or a
claim that this elementary data-span argument is new.

## Saved-data experiment, before outcomes

Bind the complete immutable inventories for:

- reference-001: manifest414e8f2b1491a7f28cd1e1522376f7b008d7d3ac57bd29f69f0d5f4b953add38;
- linear-diagnostic/attempt-001: manifestd47983793f58b033389fca5fc4aedc9ad78798b38cef06b2f3a4cf0f731534a3.

Both are below /data2/neural-shrimp-data/explore/cifar/. The first is under
device-benchmark. Load only their saved subset-A arrays. Do not open subset B,
new seeds or old LM assessment data. Hash actual arrays, source, plan and
executed environment. Refuse dirty/uncommitted executable inputs.

1. Compute reduced QR of X^T, retaining exactly512 columns, and save Q and its
   triangular factor. No SVD-based truncation, new trajectory, gradient, HVP,
   JVP, Krylov solve or spectrum. Bound orthogonality ||Q^T Q-I||_F <=1e-10 and
   representation ||X-XQQ^T||_F / max(1,||X||_F) <=1e-12. Failure is unresolved.
2. Project only the first-layer portion of every vector; leave all remaining
   parameter components in the active part. Use the five fixed saved states
   and residuals: original, and trial0/1/2/3. For each, retain full active and
   complementary parts, norms, squared-residual fractions, data-preactivation
   difference and the predicted identity (q^2-1)N.
3. Apply the same decomposition to saved p, A p and e=A p+R. Retain the
   predicted complementary identity for A p and arithmetic decomposition
   residuals. Compare identity errors with 1e-12 + 1e-8 times the norm of the
   full saved vector. These are disclosed floating checks, not exact-real
   enclosures. Use eta and lambda as the exact binary64 values in the saved
   configuration and record the rounded q and factor used in this diagnostic.
4. If all gates pass, a complementary squared-residual fraction >=0.5 at the
   original state makes an input-span solver a priority candidate. A smaller
   fraction does not explain most of this residual; report it without moving
   the threshold. This threshold is a triage choice, not a significance test.
   A high fraction alone does not prove why GMRES stalled. Never transform the
   saved diagnostic direction into an accepted Newton step in this unit.

## Resources and retention

One CPU thread, nice10; internal30s, external45s plus5s termination grace;
new raw cap64MiB including logs,1MiB finalization reserve. Exclusive new
attempt with fsynced arrays/checkpoints, caught failure outcome and manifest;
SIGKILL can leave an explicitly incomplete attempt. No paid resources or GPU.
The largest matrix is3072x512 (12MiB); avoiding a dense3072x3072 projector
bounds memory. Profile the same-shaped synthetic QR once under a committed
card before the saved target:5s internal/10s external, retain a compact timing
receipt; proceed only if2*QR_seconds+5 <=30. This is an engineering gate, not
an assurance of target convergence or runtime. Target is one fixed attempt;
no automatic retries, rank choices or extra outcomes.

Source review should include exact small-matrix identities, a nonlinear tanh
fixture with both layers trained and weight decay, q^p=1 counterexample,
rank-deficient QR completion, corrupted inventory and finite/shape gates,
state/residual ordering, cap/interruption retention and independent saved-data
recalculation. Code and an execution card must be committed before profiling
or target analysis. Current status: plan only, target not read or run.
