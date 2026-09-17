# EXP-015 audit source snapshots

These are byte-identical backups of external audit scripts used for EXP-015.
They are provenance snapshots, not production helpers or a new execution freeze.
The original absolute raw paths and script-relative exclusive output locations
are retained intentionally. Run a copy in a new external audit directory with
the original frozen source checkout available; never run these inside this
snapshot directory, overwrite old evidence, or call their checks independent
solver certification. The compact EXP-015 summary records original source and
output hashes. The metadata preflight forbids outcome-calculation APIs; the
interpretation reads saved results. Numerical arithmetic auditing is separate.

The original interpretation script is preserved without rewriting its prose.
Its broad all-phase opposition sentence has a retained erratum, and its initial
next-step recommendation is superseded by the separately dated decision note.
See `docs/EXP-015-results.md` and `docs/EXP-015-summary.json` for both bindings.
