# Fixed-R continuation from completed refinement attempts

Implementation review note, 2026-10-13. This is proposal setup only. No target model evaluation or continuation launch was performed while implementing this driver.

`center_resume.py` imports the existing pure persistence, exact binary64-to-Arb matrix conversion, residual, phase-evaluation and shooting-correction helpers from `center_refine.py`. It keeps the same fixed exact dyadic A_i and B from the original numerical seed. It adds neither Broyden nor Anderson updates and does not change proof gates. The original base plan, including all prefix, parameter, asset and seed bindings, must be preserved exactly.

## Input and source admission

The strict JSON schema is `codex-llm-center-resume-v1`, with required fields `base_plan`, `parent_attempt`, `parent_kind`, `parent_receipt_sha256`, `parent_source_sha256`, `max_corrections` and `envelope`. Optional `purpose` and `basis` fields document the run card. The parent kind is `center_refine` or `center_resume`. New corrections are an integer from one through three; the only admitted envelope is one CPU thread, nice 10, 18 GiB and 3600 seconds. The bounded launcher still supplies the actual process limits and source snapshot; standalone invocation is not an authorized launch.

`parent_source_sha256` explicitly pins the parent driver and all reused evaluation/correction dependencies listed in `COMMON_SOURCES`. For a resume parent it additionally pins `center_resume.py`. Every hash must match the retained source file and the current reviewed source. This version intentionally refuses cross-version reuse even when a change might be semantically harmless; a new importer review would be needed to relax that rule.

Admission verifies the entire parent receipt inventory, successful exit and absence of timeout, exact source commit, the parent's committed JSON plan file, backend versions, 2048-bit CPU1 context, original source lineage, pinned assets and prefix receiver binding. `base.load_bound_inputs` separately rechecks the original prefix and numerical seed receipts, full seed phase inventory, fixed binary64 matrices, parameter ratios and prefix IDs. Resume parents are traversed recursively, with cycle rejection, until a complete base refinement attempt is reached. A complete prior attempt is immutable; output ancestry is checked before creating a new output directory.

## Exact state and image reuse

Every centre is reconstructed from integer mantissa/exponent pairs with zero radius, and must remain the exact same dyadic point. No centre is converted to binary64. Every stored image retains its full radius. A reconstruction may widen an Arb magnitude; exact rational endpoint comparisons reject narrowing and do not demand radius equality.

The importer validates every phase in every completed pass: ordered complete rows, full dimensions, phase-input and output file hashes, exact input centre and cyclic successor centre, finite primal balls and exact zero tangents. The retained correction-image balls must enclose the raw encoded source images for those identical centres. For the first refinement this source is the original before pass; for continuation it is the preceding complete after pass. Comparisons use raw saved source endpoints, not an unnecessarily widened extra reconstruction.

The new attempt records a reuse manifest containing the producing attempt, receipt and source lineage; each source phase's input/output paths and hashes; original encoded images and actual decoded image balls; the exact centre tuple; and prior refinement phase counts and elapsed resources. It recomputes every residual as F(c_i)-c_(i+1) from live imported images and current exact successor centres. An old saved residual is never reused as a new correction input.

## Work and retention

Each correction persists its full before centres and residuals, the complete correction enclosure witnesses, and then the selected exact midpoint centres. It evaluates every phase at those new centres exactly once and immediately saves each complete image. Each correction has a separate numbered directory, phase rows and residual comparison. A non-improving correction does not terminate or disappear: all planned finite corrections run and their comparisons remain visible. Both a decrease in residual upper bounds and a strict residual-norm decrease (new upper bound below previous lower bound) are reported separately.

A mathematical or evaluator failure stops that attempt, preserving completed corrections and later partial inputs/images with failed markers. A launcher timeout likewise preserves whatever was written. This version refuses to import any failed or partial attempt as a parent; partial-phase import remains explicitly deferred. A completed parent is eligible only when all its planned corrections and all phase evaluations are complete.

Each additional correction needs eight fresh one-phase evaluations. Three corrections therefore need 24 phases; multiplying the measured 114.21 seconds per phase gives about 2741 seconds before setup, receipt checking and correction overhead. This is an estimate, not permission to launch or a completion guarantee. A new costed run card must use the completed first refinement's measured costs and commit/push its exact plan before any execution.

All terminal statuses include `certificate_passed: false`. Residual improvement does not establish a root, primitive period, attraction, whole-box derivative enclosure or a particular token-label cycle.

## Synthetic verification

The resume module has 40 synthetic tests; the combined resume, original refinement and shooting-correction suite passes 82 tests. Tests use two-phase, two-coordinate maps with exact offsets as small as 2^-1700 and fully mocked model construction. They cover complete base and recursive resume parents, strict plan/resource limits, source/backend/input/image-history mismatches, nonexact or missing coordinates, tangent contamination, radius widening, narrowed residual refusal, all three non-improving corrections, later partial failure, cumulative work counts and byte-unchanged parent artifacts. Whole-driver tests ensure only the fresh continuation phases are evaluated. An isolated-Python `-I --help` subprocess checks the actual launcher import path without constructing a model. No checkpoint or target model is loaded.

During development, a test fixture's unsupported deep copy of Arb values was replaced by copying the containing lists. A backend comparison initially used package metadata, which omits Torch's CUDA build suffix; it was corrected to compare the same runtime `torch.__version__` field saved by the producer. Both failures were synthetic tooling issues, not target outcomes.

A pre-review launcher check found that isolated Python excludes the script directory from its default import path. The resume driver now inserts its source directory before importing sibling helpers, and the isolated subprocess regression passes.

Independent review identified that Python dictionary equality alone admits boolean/float aliases for integer resource fields. The new resume envelope now checks exact integer types, with five regression cases; the original producer source remains unchanged.
