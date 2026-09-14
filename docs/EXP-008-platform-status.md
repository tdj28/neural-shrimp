# EXP-008 platform replay status

Private diagnostic working draft, 13 September 2026. This is not a fully
reproduced release. The exact saved experiment, numerical thresholds and
consumed attempt remain unchanged. PR7 is intentionally unmerged while its
required Linux replay fails.

## Checks at release 1d12ebeda040

The original macOS frozen replay passed all 7,008 target and 3,456 preview
records. Its independent raw audit also reconstructed the saved metrics and
features. A separate local SVD audit retained coefficient mismatches while
finding the same selected penalties and all 288 learned test actions.

Two Linux/Python 3.12.14 runs at exact commit
`1d12ebeda0400c179e54dd36e647a835f7e0d91d` failed different first checks:

| Run | First failed check | Difference | Unchanged tolerance |
|---|---|---:|---|
| [PR 34803148459](https://github.com/tdj28/neural-shrimp/actions/runs/34803148459) | Baseline standardized coefficient index 1 | Absolute 3.68269315e-8; relative 2.31758449e-7 | rtol=1e-7, atol=1e-9 |
| [Push 34803133930](https://github.com/tdj28/neural-shrimp/actions/runs/34803133930) | Shared feature index 34, `local_projected_rmse` | Absolute 1.39784018e-14; relative 3.85622692e-12 | rtol=1e-12, atol=1e-14 |

The feature values are 0.0036248908761150602 saved and
0.0036248908761290386 replayed. Although the difference is small, it exceeds
the combined declared bound and remains a failure. Full logs are retained in
[the PR log](EXP-008-linux-pr-failure.txt) and
[the push log](EXP-008-linux-push-failure.txt). These are CLI exports with
trailing line whitespace removed; timestamps, values and failure text are
unchanged.

The PR run reached fitting after the preceding raw, feature and waveform
checks. The push run stopped earlier during feature reconstruction. Neither
first-failure log is a complete mismatch census, and it is incorrect to claim
that every Linux waveform or feature check passed in both runs. The local
audit alone cannot establish Linux decision stability.

## Separate diagnostic; no replacement gate

The added platform tool reads the unchanged completed evidence, checks exact
bindings, and enumerates feature/model differences using the frozen rules.
It distinguishes refitting on saved feature rows from a complete regeneration
of those rows. It also compares saved-model decisions and refitted-model
actions. A completed diagnostic is not a pass of the required frozen verifier.
CI retains that verifier unchanged, without `continue-on-error`, and records
the diagnostic separately with `if: always()`.

The diagnostic's original-platform report is retained in
[the local report](EXP-008-platform-local.json). It passes exact bindings and
replays all 3,456 base rows (176,256 values), the saved-feature fit, all 288
learned decisions and 5,184 candidate predictions locally. Eleven focused
tests cover retained failures and rejection of changed receipts, penalties,
actions and saved-model scores. This is not a Linux result. The feature check
uses saved preview metrics and covers the 51 shared columns; it does not redo
preview phase fitting or all six geometry columns. Model refitting uses the
saved feature rows rather than a completely regenerated input pipeline.

```sh
python tools/audit_information_platform.py evidence/EXP-008/attempt-001 \
  --out /tmp/EXP-008-platform-audit.json
```

No thresholds are relaxed, no model coefficients or outcomes are replaced,
and no target trajectory is rerun. A later numerical-method change requires
an explicit new prospective packet; reanalysis of these known cases is
development evidence. The planned reduced-preview experiment must follow
that numerical work. No paid review or RunPod resource is used.

## Measured Linux decision consequences

The later diagnostic at commit `bb4ecd25cf5c025064e7bd200b6acd898953d0e5`
completed on two Linux workers. Full reports and retrieval bindings are in
[the PR report](EXP-008-platform-linux-pr.json),
[the push report](EXP-008-platform-linux-push.json), and
[their provenance](EXP-008-platform-linux-provenance.json). Both report the
expected diagnostic source hash and exact original evidence bindings.

| Quantity | PR run 34804593644 | Push run 34804589708 |
|---|---:|---:|
| Shared-feature values compared | 176,256 | 176,256 |
| Shared-feature tolerance failures | 2 | 0 |
| Baseline coefficient tolerance failures | 8 | 10 |
| Mismatched-geometry coefficient failures | 9 | 10 |
| Geometry coefficient failures | 0 | 0 |
| Changed selected penalties | 0 / 3 | 0 / 3 |
| Changed learned actions, saved-feature refit | 0 / 288 | 0 / 288 |
| Maximum refitted-score difference | 1.3874261184e-9 | 1.9615957192e-9 |
| Saved-model score replay | Pass at unchanged rule | Pass at unchanged rule |

Both enumerate all 5,184 candidate scores. The minimum best-to-second saved
score gap is 3.7010270022e-6, and every condition's gap exceeds twice its
largest observed refitted-score discrepancy. Thus the recorded action lookup
and corresponding saved-trajectory outcome are unchanged in these two
diagnostics. No new numeric pass threshold is assigned to the refitted-score
differences; exact selected-action agreement is checked separately.

The two feature failures occur at development-10, phase 675, radius 0.016,
candidate entries 16 and 17, column `local_projected_rmse`. The other worker's
feature arithmetic stays inside the original bound. Worker-dependent failure
counts are retained rather than choosing whichever platform passes more checks.

This is bounded agreement on saved inputs, not full Linux pipeline validation.
Preview metrics and certificates are saved inputs; refitting uses saved feature
rows, even when their independent reconstruction differs. The required frozen
verifier remains unchanged and is not replaced by this diagnostic's exit code.
General cross-platform reproducibility remains unresolved, and PR7 remains a
private, unmerged diagnostic draft.
