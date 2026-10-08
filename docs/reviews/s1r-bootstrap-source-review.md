# Independent S1R historical-bootstrap source review

8 October 2026 UTC. **Design review completed; implementation verdict pending.**
Reviewer basis `960ab535e1954e6f46dce21999338ddcf1bd23ee`; producer is preparing
`codex/s1r-bootstrap-association`. This record does not authorize any profile,
bootstrap trajectory, correction or saved-outcome audit.

The design in `docs/plans/s1r-historical-family-association-next.md` poses a
bounded, interpretable next question: reproduce each stated starting procedure
once, then compare the hidden-1.10 corrected pair with the saved lineage only
at identical controls after qualified continuation. The hidden-0.95 bootstrap
remains a separate finite-horizon outcome. A successful local comparison would
not recover missing historical states or prove historical/distant branch ancestry.
No material scientific design flaw was found in this read-only review.

Read the complete historical `tools/explore_a3_ns_curve.py`, `run_grid`,
`iterate_and_screen`, `GridMap` update/JVP paths, `continue_ray`, S1R constructor,
existing full-pair continuation and storage helpers, first-pilot source/outcome
reviews and the new design. The committed tracer, gdmaps, constructor and model
files have no diff between recorded historical commit `959beee68b3a` and reviewer
basis. Historical cards nevertheless identify a dirty worktree; this does not
establish those commits as the complete actual historical execution source.

Implementation checks sent prospectively to the producer and parent:

- Gradient, loss and JVP use the same pre-update state. Accepted state storage
  occurs afterward. A complete bootstrap's final tail is theta39617 through
  theta40000; its corresponding historical loop losses describe theta39616
  through theta39999. The surviving single cell makes 1,200 JVP calls: 200
  alignment and 1,000 measured growth calls. The tangent initializes with seed0.
- Durability hooks must not alter the observer-free arithmetic. Snapshot copies
  must prevent a callback from changing live states, tangents or tails. Partial
  horizons and divergence need attempted/accepted counts, last valid state,
  current frozen state and explicit indexing; partial recurrence is not the
  historical 40,000-step label. Full tail copies should occur at checkpoint or
  finalization cadence, not every iteration solely for deadline checking.
- The non-outcome timing profile should use the same single-cell
  gradient-with-loss and JVP paths as `run_grid`, resetting to the same saved
  state on every call. Cell-map gradient-only timing omits historical work.
  Its projection remains an estimate: loop, screening, checkpoint and corrector
  costs vary. No profile result authorizes a target run by itself.
- Generalized segments must consistently use their declared axis in width,
  secants and exact endpoint handling. Preserve native phase order and carried
  phase bits through direct/half/reverse witnesses. Count link attempts before
  starting another edge; 19 attempts plus six bootstrap/requalification phase
  calls use at most158 phase calls, inside the cap160. Both independent bounds
  remain hard and never turn resource stops into association evidence.
- Twelve equal macro widths are about0.00196416. Their fourth halving is below
  the minimum0.000125, so the minimum-step rule may stop before the otherwise
  permitted fourth halving. The at-most-four ceiling and minimum must both be
  enforced; diagnostic half-step witnesses have the separately inherited scope.
- Bootstrap phase seeds come from consecutive retained terminal states with
  explicit indices. Before common-parameter comparison, require identical
  binary64 log controls and derived rates. A pair distance at different controls
  proves neither association nor coexistence. Failed correction, replay,
  primitivity or spectrum checks leave the comparison unresolved.

All existing gates are exploratory numerical consistency checks. Correcting a
finite tail is distinct from proving basin membership, uniqueness or continuous
validated continuation between samples. Numeric attraction requires the complete
qualified spectra and its stated margin; residuals alone are not forward
eigenvalue enclosures. No target or profile computation has been performed by
this reviewer, and no new unit's numerical outcomes have been inspected.
