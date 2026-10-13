# Independent review of complete-parent fixed-R continuation

Reviewed 2026-10-13 by the independent Codex cascades subagent. This is a code
and synthetic-test review; no checkpoint was loaded, no target model was
called, and no saved research artifact was edited.

## Reviewed sources

| File | SHA-256 |
| --- | --- |
| `center_resume.py` | `c686ded97e5559407ab12982aec778cdc0aaf94d75bcd0e24795334ec583eda3` |
| `test_codex_llm_center_resume.py` | `4286c0a1bdec47c212a92c3714b75a8c2782222e1732442a7f9ff39c62fa9cc1` |
| `resume-design.md` | `508a1baeec31cb8c09e184d84738dfec306150bb46455e23bfed08715944f541` |
| reused `center_refine.py` | `f24f9cdecec243abd25671a684aec03b8b105288526ff7fa4f60a963af745b02` |
| launcher | `a9a4002cdd1d6e9989915fb75864935736c1387de9c167fa52d767b7678c2a1e` |

**Verdict: no unresolved blocker for the bounded proposal-only continuation.**
An actual attempt still needs its own committed, privately pushed plan with
an externally selected complete-parent receipt and source hashes. This review
does not admit a particular completed parent or authorize an extra model run.

## Admission and mathematical checks

- Every artifact in the externally pinned parent receipt is rehashed and its
  size checked. Successful exit and no timeout are required. The relevant
  saved evaluator, arithmetic, cache, correction, and driver sources must
  match both explicit plan hashes and current source bytes. Original prefix,
  seed, parameter, asset, precision, and backend bindings are checked again.
- A base parent must contain all eight before and eight after evaluations.
  A resume parent must complete every planned correction. Recursive ancestry
  rejects cycles; reuse manifests must match the admitted ancestor and every
  inherited phase artifact. Parent source changes require a new reviewed
  importer rather than merely a newly declared source hash.
- Each ordered phase binds its full exact center and cyclic successor, full
  finite image, and exact zero tangent. Missing, extra, unfinished, or
  mislabeled phase rows fail admission. Exact centers cannot lose precision.
- Decoding retains outward radii. Fraction endpoint checks prevent narrowing
  of images, correction-image history, and saved residuals. History checks
  use the original encoded endpoints, avoiding false rejection caused solely
  by an outward radius expansion on reload. New residuals are recomputed
  from the live decoded image balls; an old residual is never reused.
- Each new fixed-R correction keeps the same exact dyadic A and B proposals,
  retains the correction enclosure and exact midpoint centers, and evaluates
  all eight fresh images. Every planned finite correction is retained,
  including non-improvement. The strict residual-norm comparison uses the
  new upper bound against the old lower bound; upper-bound improvement is
  recorded separately.
- Fresh output ancestry is checked before writes. Mocked success and failure
  preserve the parent byte-for-byte. Partial work is retained on failure but
  cannot be imported through this complete-parent route. The launcher accepts
  resume only with an explicit committed plan and CPU1; its owned process
  has the existing 18 GiB and at-most-3600-second limits. Timeout is an
  incomplete attempt, never completion.

The parent source/receipt chain establishes evidence provenance, not a formal
proof that arbitrary declared data describe a model evaluation. Mathematical
validity still relies on the reviewed evaluator and pinned immutable inputs.
No interval Jacobian, shooting inclusion, primitive-cycle, attraction, or
argmax-label proof follows from decreasing these point residuals; all
certificate flags remain false.

## Verification and resolved findings

Independent CPU1 run, with bytecode and pytest cache disabled:

```text
python -m pytest -q -p no:cacheprovider \
  tests/test_codex_llm_center_resume.py \
  tests/test_codex_llm_center_refine.py \
  tests/test_codex_llm_shooting_correction.py
82 passed in 2.81s; exit status 0
```

The suite includes isolated `-I` launch/import, complete two-generation
lineage, image-history and residual narrowing rejection, mutated source and
backend rejection, all planned non-improvements, partial later failure,
exact center preservation, and immutable-parent orchestration. The original
boolean/float aliases in the resume resource envelope were identified during
review and are now refused by exact integer checks and five regressions.
The original refinement source remains unchanged. Environment stream-fd
warnings appeared before pytest; the test process completed successfully.
