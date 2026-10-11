# Root source review of the broader-data LM audit

8 October 2026 UTC. Reviewed auditor source/card
`f773b70e053612f2e91d88cb4d506c7d29b50753` before its target execution.
Root read the complete audit, prospective card and synthetic controls. The
independent arithmetic uses augmented least squares rather than the producer's
normal equations, direct corpus-token reconstruction, explicit causal AR
designs and all candidate/eligible-row comparisons with exact gate decisions.
The fixed feature replay uses shared reviewed primitives; its eight samples
and cross-device tolerances are disclosed separately from complete arithmetic.

Root's integrated CPU1/nice10 synthetic check passed 29 tests in 6.81s:
`tests/test_tiny_lm_broader_audit.py` plus `tests/test_tiny_lm_progress.py`.
An earlier invocation named a nonexistent broader-data test file and exited
before collecting any tests; the corrected command above is the executed set.
No target artifact was opened by these tests, and no tolerance was changed.

No blocking provenance, causality, numerical-method or bounded-execution issue
was found. Limitations remain: the saved Ritz estimate is bound, not recomputed;
full state transitions and all losses/gradients are not exhaustively replayed;
probe observations are summarized without new probe forward evaluations;
feature/model primitives are shared; native calls rely on the external timeout
in addition to the signal deadline. Sample failure makes the audit fail, while
time exhaustion leaves it incomplete. Neither authorizes a relaxed threshold.

Authorize one execution of the prospective card after a fresh shared-resource
check, CPU/BLAS1/nice10, internal60s/external90s plus5s, raw32MiB, exclusive
`/data2/neural-shrimp-data/audits/tiny-lm-broader-data/attempt-001`.
Pin collection producer/card commit
`44ed5ef60bdb300087c04d5a7b7d3fd7faf75dee`, collection manifest
`2ec1948521fe50a05a81b0ddec2e3e741cb25745c58f887868e3c5db43e2413d`,
analysis SHA256
`a84558767147a0d9521e4d9f597982075145d96e36c4dffabe1039b7536179c3`.
Use the card's exact collection, analysis and original corpus paths; no GPU,
HVP, trajectory updates, probe reevaluation, paid compute or follow-on.
Preserve and report every stage and any failed/incomplete attempt, then stop.
