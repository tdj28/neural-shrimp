# Synthetic final-gate resource pilot

Prepared only; no launch or actual seed admission by the implementation agent.
CLI: `gate_budget.py --output OWNED/new-attempt/run --plan captured-plan.json`.
The launcher requires a committed plan, one CPU thread and exactly 1,800 seconds;
the existing 18 GiB cap, nice10, process ownership and timeout receipt apply.

The complete numerical seed receipt and captured source inventory are externally
pinned. Only NPZ members A and B are decoded; original centres, images, product,
residuals and inverse-defect arrays are not decoded. Source-byte authentication
is distinct from Git blob provenance verification. The base A/B arrays are
copied to a new immutable artifact and their source/raw hashes retained.

Each synthetic exact proposal entry is the stored binary64 A entry plus
`(-1)^(phase+row+column) * 2^-1000`, constructed using exact integer ratios.
Every perturbed mantissa must fit in 1,024 bits and remain an exact Arb point
through matrix assignment; the driver refuses instead of rounding a proposal.
B is the exact dyadic interpretation of the saved binary64 B. Synthetic J_i
has that perturbed proposal centre and radius `2^-1100`, with outward storage
and finite/containment checks. Synthetic phase i is the vector with every
coordinate 4i, nominal radius `2^-800`, and its image is exactly the next
synthetic phase centre. These data describe local affine maps on disjoint
synthetic phase neighborhoods, never the Qwen soft-feedback map.

Perturbing A as well as J makes the factorized root products high-mantissa.
This is a conservative engineering stress workload, not a proved upper bound
or exact runtime forecast for the eventual target gate. It calls the unchanged
shooting wrapper once, with `bound_method='factored'`, `powers=(1,2)`, 1,024-bit
arithmetic and one thread. Both a conditional pass and a conditional failure
are completed engineering outcomes. Any flags are nested under explicit
`synthetic_only=true`, `certificate_passed=false`, with model evaluations zero.

The driver preserves backend versions before admission, stage events with
elapsed time and peak process RSS, copied base proposals, per-phase construction
and hash records, exact synthetic centres/images/radii, the deterministic recipe
and all matrix/contract hashes. It saves the compact gate summary before
serializing the full wrapper report. JSON encoding is streamed to a `.partial`
file and renamed only on success; an interrupted partial is kept. Owned SIGTERM
and ordinary exceptions attempt to save a failed status; if native arithmetic
delays signal delivery, the outer launcher's timeout/kill receipt and previously
saved stages remain the failure evidence. Tighter inherited address-space caps
are preserved.

At p8/n896 the unchanged wrapper path has 35 dense matrix multiplication calls,
in addition to construction, matrix-vector work, validation, norms and hashes.
Its attraction report contains 2,408,448 ball dictionaries. Roughly 29 live
native matrices would carry 2.78 GiB of full 1,024-bit midpoint bit payload alone;
this excludes radii, native structures, Python objects and matrix workspaces.
The 18 GiB and 1,800-second limits are exploratory hard ceilings, not established
performance bounds. Root must admit simultaneous jobs against current memory
and CPU availability before launch. No GPU or paid compute is requested.

Validation uses only two phases of dimension two and synthetic NPZ archives.
The archive's unused members deliberately require pickle, so successful loading
with `allow_pickle=False` demonstrates that they are not decoded. Tests include
exact long-mantissa construction/refusal, original A/B preservation, real tiny
full gate and serialization, a retained failing conditional gate, complete and
failed CLI orchestration, synthetic SIGTERM retention, tighter inherited caps,
source/plan/input tampering and launcher admissions/refusals before any Git call.
