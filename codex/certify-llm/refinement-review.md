# Independent review of the bounded centre-refinement setup

Reviewed 2026-10-13 03:31 UTC by the independent Codex subagent. This is a
source/synthetic-test review, not a target evaluation or a certificate. The
reviewer loaded no checkpoint and evaluated no target-model phase.

## Reviewed versions and test evidence

SHA-256 hashes at review:

| File | SHA-256 |
| --- | --- |
| `center_refine.py` | `f24f9cdecec243abd25671a684aec03b8b105288526ff7fa4f60a963af745b02` |
| `shooting_correction.py` | `06a711c7becd970bb135bc28555bf91dfd24bd52a7c224f5ea2d25647f8bac1e` |
| `cache_io.py` | `f4e25b2134bef8a71c27d3156baf01be9d25c593d7b780f90a66b9c1141abdef` |
| `tests/test_codex_llm_center_refine.py` | `76847078d14ad2aed81b13e1848482e262bbcc9c09b079c95062c16f6c989951` |
| `tests/test_codex_llm_shooting_correction.py` | `824f74b08eeb895163817bcb303f90a7f0161a647632d986e43475eedd84519e` |

Independent command, with OpenBLAS/OMP/MKL each restricted to one thread and
bytecode generation disabled:

```text
/data2/neural-shrimp/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_codex_llm_center_refine.py \
  tests/test_codex_llm_shooting_correction.py
```

Result: **42 passed in 1.33 seconds**, process exit status zero. This includes
an eight-phase synthetic orchestration test with the checkpoint loader and
map replaced by tiny test doubles, exactly one correction and 16 point calls.
It also checks high-precision midpoint retention, failure retention, bad
upstream receipts, changed input hashes, parameter/scope admissions and the
correction against an exact small nonnormal cyclic system. The synthetic
orchestration test does not by itself validate real upstream artifacts; the
actual plan's receipts must still pass the driver's admission checks.

## Mathematical review

No blocking mathematical issue was found in the reviewed source.

The stored binary64 `A_i` and `B` are converted to exact dyadic Arb entries.
They are fixed preconditioner proposals; neither numerical inverse defects
nor the one-phase floating Jacobians are called rigorous derivative bounds.
For `h_i=F(c_i)-c_(i+1)`, the recurrence uses

```text
s_0 = 0; s_(i+1) = A_i s_i + h_i;
delta_0 = B s_8; delta_(i+1) = A_i delta_i + h_i.
```

These signs implement `delta=-R h` for the implicit shooting preconditioner.
If `B` were the exact cyclic inverse, they would solve the linear correction
system. The code does not assume that identity. It retains the cyclic defect,
residual balls and correction balls and deliberately selects exact midpoint
centres only as proposals. The second complete point-evaluation pass recomputes
residuals at those new centres.

Both upper and lower rigorous infinity-norm bounds are retained. A decreased
upper bound is labelled separately from the stronger comparison
`after_upper < before_lower`, which establishes an actual residual-norm
reduction. Neither comparison establishes a root or stability.

## Provenance and retention review

The fixed plan admits the assigned period-eight, 896-dimensional map,
`alpha=0.071674`, `log10_temperature=-0.115`, 2048-bit arithmetic, exactly one
correction and exactly two evaluation passes. It binds upstream receipts,
map/seed source hashes and the final preconditioner NPZ hash. The loader checks
all inventoried upstream files, source-commit bindings, checkpoint/config/
steering hashes, prefix IDs, exact parameter ratios, completed setup statuses,
all eight saved numerical phase gates and the complete seven-array finite
binary64 inventory. The prefix cache is imported through the previously
reviewed receiver-bound cache loader, not accepted merely by path or label.

Each phase's exact input is saved before evaluation; each successful full image
is saved immediately. The correction witnesses and new exact centres are saved
before the second pass. Failures preserve earlier images and the failed phase's
input. A failure cannot be reported as a completed 16-phase attempt. Hashes
bind provenance, while enclosure correctness still depends on the reviewed
Arb evaluator and its actual successful execution.

## One-hour bound and result interpretation

This is admitted only as a **CPU1, 18-GiB, one-hour setup attempt**, at nice 10,
through the separately committed, privately pushed bounded launcher. The
review grants no execution permission beyond the parent's authorized run card.
The driver intentionally performs no root, whole-box derivative, primitive
period or attraction proof gates and writes `certificate_passed=false` even
when both passes complete.

One hour is an enforced execution ceiling, not a predicted completion time.
If it expires mid-pass, the launcher's timeout/nonzero-exit receipt and retained
phase artifacts must be reported as an incomplete setup. A smaller residual,
a completed correction, or a partial after-pass must not substitute for a
successful complete attempt. Even a fully completed refinement is only input
preparation for later interval existence/uniqueness and attraction checks.

## Launcher integration

The updated launcher was reviewed at SHA-256
`7b4c22fe16e24e8c3e22acd0eee1319d3292276fe518b14e8d376bb1069ce245`.
Its `center_refine` integration is approved for the bounded setup scope above.
It requires a named committed JSON plan, rejects path traversal and uncommitted
source, verifies the source commit is on fetched private `origin/main`, and
passes the plan from the captured source snapshot. Only `center_seed` can use
more than one thread; refinement retains CPU1, nice 10, 18-GiB address-space
limit, a maximum 3600-second wait and owned-process-group termination with a
10-second termination grace before SIGKILL. The grace is cleanup time, not
permission to continue research after the deadline. The attempt manifest and
final receipt record source hashes, ownership, exit status and timeout.

Four invalid refinement launcher configurations were independently checked:
missing plan, two requested CPU threads, a traversing plan path and `--skip-box`.
All four failed before Git access or attempt creation. The valid launch path
was reviewed without executing it. No target run was made as part of this
review. A timeout still requires an incomplete result regardless of which
intermediate files happen to exist.
