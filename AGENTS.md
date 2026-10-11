# Research and repository policy

The human requested this separate project on 2026-09-08. Keep the GitHub
repository private. Creating and pushing this private repository is authorized;
public release, external messages, and paid review calls are not.

Read the canonical research playbooks in `../agent-skill-documents/` when
available: `EXPERIMENT_INTEGRITY_SKILLS.md`, `GPU_COMPUTE_SKILLS.md`,
`GIT_WORKFLOW_SKILLS.md`, and the applicable parts of `RESEARCH_NOTE_WRITEUP.md`.
This repository records their hashes in `docs/skill-provenance.json`.

Read [Skills.md](Skills.md) for the current server and shared-resource policy.
The user authorizes local CPU, RAM and GPU use on `zeta`; prefer suitable
local resources and check availability without disturbing other agents.
RunPod remains an acceptable fallback when local capacity is insufficient or
the GPU is busy, subject to the paid-run requirements below. This local
amendment supersedes older weak-computer assumptions in the playbooks.
Checkpoint and push code regularly as specified in Skills.md. Keep new bulky
raw data under `../neural-shrimp-data/`, with compact hash-bound receipts in
Git; preserve existing frozen evidence and history.

- Initial experiments are exploratory pilots, never formal preregistrations.
- Follow [the adaptive research protocol](docs/adaptive-research-protocol.md)
  (user directive, 6 October 2026): explore freely with code committed and a
  run card in the research log before each run; adapt any plan when findings
  warrant it through dated, disclosed amendments; push a frozen test design
  before outcomes only for confirmatory tests of headline claims on fresh data.
- Preserve every failed or incomplete attempt, immutable raw evidence, exact
  source commit, manifest hashes, and changes of plan in a dated amendment.
- Do not label visual islands or loss oscillations as verified shrimps. Verify
  full-state primitive periodic orbits, stability, and boundary organization.
- Utility of cycle detection is distinct from incremental utility of geometry.
- Before claiming a shrimp-specific behavioral effect, test comparable distant
  regimes, including both chaotic and ordinary periodic controls. Qualify
  chaotic controls with explicit chaos diagnostics; retain unresolved or
  finite-horizon irregular cases under their actual evidence labels.
  An effect that also appears
  away from the shrimp remains generic unless the geometry adds demonstrated
  explanatory or predictive value. See docs/EXP-027-status.md for the paused
  control-design requirements requested by the user on 23 September 2026.
- Use local audits. Paid Pro reviews are optional at major milestones and
  require explicit approval for each call under canonical integrity section 7A.
- Paid RunPod execution requires a measured, costed run plan, an approved hard
  ceiling, exact task resource ownership, artifact retrieval and verified
  termination. A prepared archive does not authorize a launch.
- Work directly on `main` during the early research stage (human directive,
  11 October 2026). Commit tested changes and push them to `main` regularly;
  pull requests are not required. Use a short-lived branch only when parallel
  agents need isolation, and merge it back into `main` promptly. Do not leave
  long-lived or divergent branches. Stage files explicitly and sweep staged
  content for secrets. Never force-push or otherwise rewrite published
  history: receipts cite commit hashes. This supersedes the branch-and-PR
  workflow of the canonical Git playbook for this repository.
- Do not add assistant coauthor trailers to commits.
