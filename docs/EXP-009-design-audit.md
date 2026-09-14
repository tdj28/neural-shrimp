# Numerical follow-up design and implementation audit

14 September 2026, before the first regenerated-input audit and EXP-009
integration. The separate arithmetic diagnostic has its own earlier pushed
freeze and review. All three investigations use known EXP-008 outcomes for
numerical development and preserve the original evidence and failing gate.

## Saved-trajectory regeneration

Reviewed `tools/audit_information_regeneration.py` against the original
driver and scoring routines. It rebuilds sensitivity readouts, Grams,
preview waveform metrics, shared and geometric features, development labels,
models and test choices. Fitting receives regenerated development inputs;
it does not silently fall back to saved features or labels. The original
sealed-action, label-only and regenerated-action outcome panels distinguish
different consequences of regeneration.

Saved trajectories remain attached to their original realized actions. A
fresh eigendirection is compared separately and cannot redefine the action
that generated the stored trace. Equivalent phase minimizers are diagnosed
without replacing the original saved-phase replay rules. This is a full
saved-trajectory analysis, not fresh ODE or variational integration.

Independent review found that matching NaN/infinity values could be converted
to strings before comparison and erroneously pass. This was fixed before
outcome execution; nested-array regression tests confirm nonfinite failures.
Missing rows, binding failures and interruptions retain their status and
cannot produce a complete primary result. Source and evidence are checked
again after the run. The source checkout must therefore stay clean until the
audit ends.

## EXP-009 integration refinement

The machine plan reproduces all 3,504 old pairs and the fixed 1,369-case union,
including all 384 policy assignments to 220 distinct selected trajectories.
Its deterministic selection hash, source bindings and old raw input hashes
validate. All selected saved initial states, references, epochs, parameters
and common sample times were checked without new integration.

Independent review confirmed the two native steps and the fixed 19-case,
two-setting DOP853 panel, phase units/sign, all-required policy denominators,
external-output guard and failure retention. Two implementation findings were
fixed before execution: process ownership now covers interruption during
launch, and replay checks the entire phase/window/component/cadence score
tree as well as aggregate values. Synthetic launch interruption confirms
that no task-owned worker survives cleanup. Only explicitly owned processes
are terminated. This runner is qualified for the current Linux platform.

The sampled cohort and its controls are outcome-selected numerical cases.
They do not estimate convergence prevalence across unselected candidates.
Full-state disagreement, RMSE disagreement and fixed-action policy effects
remain separate outputs. No incomplete candidate subset is used to calculate
a new oracle, prediction MSE, model fit or regret. Further integration levels
are not automatically authorized by an unresolved result.

## Validation and freeze

The combined source passed **325 tests in 21.97 seconds** using one thread per
numerical library. The two warnings concern optional PyYAML formatting of
NumPy configuration output; no dependency change was needed. The refinement
plan's `validate` command independently regenerated its complete prior-data
selection and checked all bound inputs/sources. `git diff --check` passed.
Local review found no remaining blocking finding.

The complete new code, tests, methods and machine plan must be committed and
pushed before the two runs. The actual remote branch SHA will be checked;
EXP-009 also performs that live check itself. Full outputs stay under
`/data2/neural-shrimp-data/`, with compact hash-bound results retained in Git
after execution. CPU workers are capped at four and each numerical library
at one thread. No paid resource or review is used.
