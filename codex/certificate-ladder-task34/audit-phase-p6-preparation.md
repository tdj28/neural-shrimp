# Explicit cascade-P6 saved-phase auditor preparation

Prepared in `/tmp` only, 2026-10-18. Conditional source/interface approval for
Root's integration and separate bounded audit admission after original closure.
No actual P6 output, receipt, admission, model asset or other raw evidence was
read. No model/target call, Git action, host/process query or active-source edit
was performed. The scientific phase run remains Root-owned.

## Why a separate auditor is required

The reviewed `codex/certificate-ladder-task34/audit_phase.py` executes its audit
on import and fixes its original source to
`32c7e4126cd88c6e0c2030dcf1d287f356ee2db8`, raw directory and completion path to
`phase-cascade-p3-001`, cell to `cascade-p3`, period to 3, and stacked matrix
shape to `(3,896,896)`. It is not a parameterized P6 auditor. The new
`/tmp/audit_phase_p6.py` performs no evidence access on import and is explicitly
bound to the original `phase-cascade-p6-001` producer source
`36366f7fb9b81e5f90d61aee1d09ae5730dff666`.

## Scope and source/interface review

The wrapper requires the exact original shared five-cell
`phase-cascade-p3-001-card.json`, P6 plan, original successful phase receipt and
admission, original committed scientific source inventory, one-thread limits,
all original card input pins, and original supervisor/worker closure. The cell
is selected by `cascade-p6`, including exact binary64 parameter ratios, original
P6 history/report paths and p3 seed authority. A different period-6 cell cannot
substitute for this cell. Receipt and admission hashes are mandatory explicit
arguments; a passing report JSON alone cannot establish this authority.

There must be exactly fourteen original output files: report, proposal archive,
six phase matrices, and six phase residual archives. The report must have one
exact named-cell row and ordered phase indices 0 through 5. The seven proposal
arrays must be finite binary64 and have their complete expected shapes:
`A`, `B`, `centers`, `parameters`, `product`, `left_inverse_defect`, and
`right_inverse_defect`. Centers must equal the last six rows of the original
513-by-896 history. Archive duplicate fields, omitted/extra arrays or phases,
and altered original pins are refused. The ordinary runtime fixes dimension
896; only the isolated array helper accepts a small fixture dimension.

Numerical formulas are unchanged from the reviewed P3 auditor: each per-phase
matrix equals its stacked entry; each saved center matches the phase center;
residual equals image minus the next center, with phase 5 wrapping to phase 0;
each saved relative L2 closure is recomputed and must be below `1e-10`.
The ordered product is `A[5] @ ... @ A[0]`; the saved product is compared exactly.
With `D=I-product`, both saved defects are recomputed as `I-B@D` and `I-D@B`.
The four infinity-norm diagnostics and `numpy.linalg.cond(D)` are compared
exactly. No inverse solve, Jacobian, derivative sampling, model replay, or new
scientific experiment is performed. `B` remains a finite proposal; matching
saved arrays does not establish a rigorous inverse or derivative enclosure.

The wrapper checks original file pins before and after array work and checks
original process/group absence twice using the pinned supervisor helper.
Output is an exclusively created separate
`phase-cascade-p6-001-audit-001/result.json`. A runtime refusal is retained in
that fresh directory's `failure.json` with traceback, source, cell, period and
requested original receipt/admission pins. Original evidence is not edited.
Successful and failure outputs both retain `certificate_passed=false`.

## New manufactured controls

One run of `/tmp/audit_phase_p6_controls.py` completed with exit 0: **45 passed,
0 failed**. The controls cover new P6 source/card/command/history/period joins,
fourteen-file authority, six ordered phase rows, seven-array and final-phase
interfaces, mocked original closure, explicit argument pins and fresh refusal
retention. The whole runtime loader is exercised with mocked raw file, Git,
process and numerical-checker interfaces. A separate dimension-2 six-phase
fixture exercises the unchanged array checker and cyclic phase-5 relation.
Its matrices are small manufactured binary64 proposals, not target evidence.
An audit hook blocks raw/model-file opens, `/proc` reads and subprocess actions
inside the test process. Source/card/plan files alone are read from the repo.
No earlier scientific computation or prior control suite was repeated.

Results are `/tmp/audit-phase-p6-controls-attempt001.json`; stderr is empty.
The manufactured failure and unchanged sentinel are retained under
`/tmp/phase-p6-audit-controls-9ug0cxh_`. There was no failed test attempt.

## Conditions before actual use

Root must integrate and independently admit the exact auditor under a separate
bounded saved-audit card/plan, after the original run has closed. Populate the
mandatory `--receipt-sha256` and `--admission-sha256` from that closed original
allocation, with `--attempt phase-cascade-p6-001`, the exact original source,
`--cell cascade-p6`, and the fixed separate result path. No original runtime
pin or outcome is inferred in this preparation. Parent directories and any
failed audit allocation must remain preserved; an occupied audit-001 path
requires a disclosed new preparation, not overwrite or automatic retry.

The actual audit needs host-visible original process/group closure, resource
admission, and review of its own completion receipt. The source-pinned helper
can only report processes visible in its execution environment; isolation must
not be treated as proof that host processes ended. Exact numerical replay can
refuse if the NumPy/BLAS backend differs. Preserve such a refusal and resolve
backend identity explicitly; do not relax equality. The audit records Python,
NumPy and thread settings, while backend/resource qualification remains part
of Root's bounded admission.

Any successful audit certifies only consistency of authenticated saved finite
proposals with the reviewed original execution and their saved algebra. It
does not independently establish the model-derived Jacobians, rigorous orbit
closure, primitive period, trajectory, or a certificate.

## Exact pins

| File | SHA256 |
| --- | --- |
| `/tmp/audit_phase_p6.py` (19920 bytes) | `61c8a466235eedbfcc9454d5965e8e999a46a88a79552c3ab7e799f7f2aaa92e` |
| `/tmp/audit_phase_p6_controls.py` (16513 bytes) | `dcf3d6b673ac18ae8d1b81e7f1ad5bc921bd8bdc1d77372489b3bb7a21251c3f` |
| `/tmp/audit-phase-p6-controls-attempt001.json` (9503 bytes) | `79fc93877866b2cd83d57d24a4a6a05a864a06d7419fd7adb152e1080ef31de4` |
| `/tmp/audit-phase-p6-controls-attempt001.stderr` (0 bytes) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Existing `audit_phase.py` | `c781f0474897e40ea47832db2cefd9cf22027c637cc7b6db3f9198f3d46890ac` |
| `phase_preconditioner.py` | `71e3a6046a4f0857df83089a508c1bad49851c00145abb81713bf7227f6c106f` |
| `derivative_profile.py` | `63d17376d9d33a7fda7669dfba03f317013a0aee19564533280edf9c51fbaf6b` |
| `center_seed.py` | `87cf27a05999641995d441704cd7a1ec3ea5aa0d2c0d200b0c05662e584572e5` |
| `qwen_map.py` | `493a3d1d57fec227f5ba2e3f8f9cfcfdd02d22495d1d44e6449ede901310ae6e` |
| `supervise.py` | `306db32a2c8ada9beb67547bbaafc6d0d7a597b5861b0e629ca9a07801223c44` |
| Original shared `phase-cascade-p3-001-card.json` (6064 bytes) | `fad1beeb1eda8afaed84bcf315d27ffc1f1e446f3c2694b9690c1c702e87aa4a` |
| Original `phase-cascade-p6-001-plan.json` (10307 bytes) | `07713794955b1a7ca18de10ff8b0038ff4dd8eb9eb398c7967a1cebae2322177` |

## Root disposition 2026-10-19T03:16:50.481497+00:00

Root read the entire isolated source and review. Preparation and45manufactured controls are retained without a target audit. The exact original phase-cascade-p6-001 subsequently closed incomplete at its RAM floor, with one residual and no complete matrices/proposal. Its success/14file gate necessarily refuses this attempt; do not run it to manufacture another expected refusal. No audit runtime is admitted. Any separately disclosed future scientific attempt would need an explicit new source/allocation adapter and fresh audit admission; this source cannot silently be relabeled. Original001closure is already authenticated in phase-cascade-p6-001-completion.json. The preparation header date was agent-local2026-10-18; integration and this disposition are actualUTC19October.
