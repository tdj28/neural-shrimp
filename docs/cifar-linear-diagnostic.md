# CIFAR saved-state linear-direction diagnostic

8 October 2026 UTC. Exploratory rung-0 numerical diagnosis of the previously
unresolved corrector. This is one saved subset-A state and one unchanged Krylov
budget, not another training run, an accepted orbit, or a continuation result.

Source `a9830e714d259597a53cee6ef7be41a9f07bb205` and the dated prospective
card were committed and pushed before the run. Independent source review
`dc7b4fea57c26c0a9e6db1b255c0e1d44efc1799` passed, including the full-budget
synthetic count of 200 inner iterations and 205 operator calls. Root authorized
this exact CPU-only attempt after review. The original unsuccessful corrector
and every diagnostic trial remain preserved.

## Result

The same 50-by-4 GMRES solve again returned `info=4`, with no linear convergence
at the requested relative tolerance 1e-10. Its estimated relative residual
started at 0.899404945 and ended the four restart cycles at 0.892172055,
0.888564174, 0.886202741 and 0.884313632. Independently applying the operator to
the returned direction gave true unpreconditioned forcing ratio
**0.8843136324964805**, above the frozen 0.5 limit. The final estimate agrees
with the true check to about 5e-15; a misleading internal residual estimate is
not the explanation for this failed solve. The residual history shows slow
reduction under this budget; it does not determine a condition number or prove
that a nearby periodic orbit does not exist.

The direction has negative residual-merit derivative, -4.6880896636e-11, but
its full-step relative norm is about 0.03888. All four independent dampings
were evaluated from the original state. Merit is the absolute squared closure
residual divided by two; changing the state normalization cannot manufacture
merit improvement.

| Damping | Merit / initial merit | Relative closure | Armijo | Frozen inexact-step candidate |
|---:|---:|---:|:---:|:---:|
| 1 | 21.3925568722 | 2.3006225590e-5 | fail | no |
| 1/2 | 2.1372607360 | 7.2724291133e-6 | fail | no |
| 1/4 | 0.9816889639 | 4.9282697878e-6 | pass | no |
| 1/8 | 0.9539323048 | 4.8576844290e-6 | pass | no |

The quarter and eighth steps decrease merit by about 1.83% and 4.61%, but all
four fail the unchanged forcing rule. All four also fail the scientific
closure threshold 1e-8, versus initial closure 4.9730550595e-6. Their one-step
separations remain above 1e-7. Passing that separation screen alone establishes
no periodic orbit. There are **zero qualified candidate steps, zero accepted
orbits and zero spectra**. No trial was chained, applied to continuation or
used to adapt the forcing threshold. A later choice to allow weaker linear
accuracy would be a new disclosed algorithmic amendment, not a reinterpretation
of this result.

## Evidence and resources

`docs/cifar-linear-diagnostic-receipt.json` binds the exact source and source
review, both immutable input manifests, normalized subset-A identity, the
original state hash, runtime libraries, operation counts, full trial metrics,
process status and output manifests. Raw data are in
`/data2/neural-shrimp-data/explore/cifar/linear-diagnostic/attempt-001`.
Returned direction, its applied operator, true linear residual, all four trial
states, both mapped phases and nonlinear residuals are retained. All 32 files
(including the manifest) total 18,218,537 bytes and are read-only. Producer
inventory, size and SHA-256 rechecks passed. An independent saved-outcome audit
is pending; no target derivative or training replay is needed for an arithmetic
check of the saved residuals and declared gate.

Initial closure/separation replay passed. The stored gradient, one HVP and one
J2VP were bit-identical to the previous CPU reference. Process exit was 0;
wall time was 10.6354 s including interpreter startup, with 9.1886 s inside
the diagnostic, 8.0113 s in the solve/true-check block and maximum RSS 787,500
KiB. CPU/BLAS and Torch interop were one thread, nice 10. The internal 100 s,
external 115 s plus 5 s kill grace and 32 MiB output limits were not reached.
No GPU, paid resource, subset B, new seed, Arnoldi spectrum or continuation
was used. The sole runtime warning was the existing Torch `jit.script`
deprecation warning.
