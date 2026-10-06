# Resume the shrimp research on a new server

Prepared 13 September 2026 (America/Los_Angeles). This is an operational
handoff of the private research at `ba4df5357beaf816a466e6e5c164d631ba34bb5e`,
not a new experiment, release, or numerical-method amendment.

## Recommended: preserve this exact conversation with SSH handoff

Keep the desktop app as the interface and run the work on your new server.
OpenAI documents SSH projects and transferring an existing chat and its Git
state in the [remote connections guide](https://learn.chatgpt.com/docs/remote-connections#connect-to-an-ssh-host).

1. Establish normal SSH access to the server using a concrete alias in your
   local `~/.ssh/config`. Install and authenticate Codex on the server, and
   ensure `codex` is on the remote login shell's PATH. Follow the current
   [official CLI installation instructions](https://learn.chatgpt.com/docs/codex/cli).
2. Clone the repositories below onto persistent server storage. Recreate the
   Python environment there. Authenticate GitHub separately on that host;
   both the research and skills repositories are private.
3. In the desktop app, use **Settings > Connections > SSH** to add the host.
   Save the remote `butterfly` folder as a project matching the current local
   `butterfly` project. Also save `neural-shrimp` as a remote project.
4. After this response finishes, select the run location in this chat's footer,
   select the server, review the destination/branch and choose **Hand off**.
   This transfers the chat and Git state to a remote worktree. The current
   chat cannot call its own handoff tool; the footer can perform it.
5. Tell the resumed chat the **absolute server paths** of `neural-shrimp` and
   `agent-skill-documents`, and have it read this file before doing research.
   A managed `butterfly` worktree may not have those repositories as siblings.

**Project mismatch matters:** this conversation is attached to
`tdj28/butterfly`, although the shrimp work is in `tdj28/neural-shrimp`.
Moving the `butterfly` chat does not substitute for cloning the separate
shrimp and skills repositories. The app requires a matching saved Git project
on the destination to offer it as a handoff destination. At preparation time,
only the local host was configured; no server connection or transfer has yet
been performed.

If you will retire the desktop computer too, use Codex CLI directly on the
server and paste [the restart prompt](docs/handoff/restart-prompt.md). This
preserves research continuity through the packet; it does not itself import
the old conversation. Keep the original chat until the transfer is checked.
`codex resume` resumes a session available to that Codex instance; a Git clone
does not copy chats. See the [CLI command reference](https://learn.chatgpt.com/docs/developer-commands?surface=cli#codex-resume).
Do not copy the macOS virtual environment or wholesale authentication/config
directories to make a Linux installation.

## Clone the actual current work

Choose a persistent parent directory on the server. The example uses
`~/PROJECTS`; run these commands there, after GitHub authentication. They
assume the destination folders do not already exist.

```sh
mkdir -p ~/PROJECTS
cd ~/PROJECTS
gh repo clone tdj28/neural-shrimp -- --branch codex/shrimp-information-ablation
gh repo clone praxagent/agent-skill-documents
gh repo clone tdj28/butterfly -- --branch codex/exp525-result-audit
git -C neural-shrimp fetch origin --tags
git -C neural-shrimp merge-base --is-ancestor ba4df5357beaf816a466e6e5c164d631ba34bb5e HEAD
git -C neural-shrimp status --short
git -C agent-skill-documents switch --detach 053438a0afddcd10b5000de1cdc227dc5050324b
```

Use a full clone with history and archive tags. `main` at the snapshot is
`16abe0d84496126f1a651a29797ba7859de477f5` and stops at EXP-007. The archive tag
`archive/EXP-008-source-v1` is the earlier source freeze, not the current
manuscript/results. [PR #7](https://github.com/tdj28/neural-shrimp/pull/7) is
open, draft and intentionally unmerged. Do not merge it to simplify migration.

The pinned skills checkout reproduces the current canonical policy; it does
not authorize editing or executing an older paid-review client. All four
playbook hashes still match `docs/skill-provenance.json`. Check later policy
updates explicitly rather than silently adopting or ignoring them.

For this shrimp project, the full research clone contains all essential
evidence: 23,451 tracked evidence files, approximately 1.6 GB of evidence,
plus manuscript, code, figures, plans, receipts and failure records. Allow
additional space for Git history, dependencies and future runs. No Git LFS,
submodule or external symlink is required by the inspected snapshot.
The canonical checkout has no untracked work; ignored virtual environments,
bytecode and temporary PDF-rendering images can be recreated. The `butterfly`
clone is for same-chat project matching; this packet is not an audit of that
separate project's ignored datasets or experiments.

## Recreate and check the environment before new outcomes

Use CPU float64 first. The recorded reference used macOS ARM64, Python
3.13.11, NumPy 2.5.3 and SciPy 1.18.1. Linux CI used Python 3.12.14 and the
same locked scientific package versions. More RAM or a GPU does not resolve
floating-point portability. GPU execution would be a separately validated
implementation; this project's present NumPy/SciPy reference runs on CPU.

On a server with Python 3.12 and its venv module installed:

```sh
cd ~/PROJECTS/neural-shrimp
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps -e .
python -m pip check
```

This mirrors the existing CI install. Record the actual Python patch release
and build environment; the dependency lock is not a bitwise OS/BLAS/container
freeze, and the editable package's build isolation can resolve build tools.
`uv.lock` is also preserved. Do not regenerate either lock during migration.
If pinned packages cannot be installed on the chosen architecture, retain
the failure and select a documented compatible environment before proceeding.
Native experiment execution also needs a C++17 compiler. PDF reading needs
no TeX installation; rebuilding the manuscript also needs
LaTeX/BibTeX and Poppler as described by the existing paper build instructions.

Create a **new, persistent** audit directory, outside the evidence tree. Use
a fresh directory for each attempt, with distinct stdout, stderr and exit
status records. Inspect the existing immutable evidence before generating
anything new. Record CPU/OS, Python, pip freeze, NumPy/SciPy build information,
BLAS/thread settings and, if present, GPU/driver details. Do not dump credentials
or the entire process environment into a report.

The following read-only checksum check needs only Python's standard library.
Run from the repository root. It checks the handoff bindings, all 178 manuscript
provenance entries, all 10,650 EXP-008 receipt artifacts and canonical playbooks.
It does not numerically replay trajectories or establish scientific convergence.

```sh
python - <<'PY'
import hashlib, json
from pathlib import Path

root = Path.cwd()
def check(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'SHA-256 mismatch: {path}')

snapshot = json.loads((root / 'docs/handoff/snapshot.json').read_text())
for entry in snapshot['critical_files']:
    check(root / entry['path'], entry['sha256'])
provenance = json.loads((root / 'paper/provenance.json').read_text())
for entry in provenance['files']:
    check(root / entry['path'], entry['sha256'])
attempt = root / 'evidence/EXP-008/attempt-001'
receipt = json.loads((attempt / 'receipt.json').read_text())
for entry in receipt['artifacts']:
    path = attempt / entry['path']
    if path.stat().st_size != entry['bytes']:
        raise SystemExit(f'Size mismatch: {path}')
    check(path, entry['sha256'])
skills = root.parent / 'agent-skill-documents'
for name, digest in snapshot['skills']['files'].items():
    check(skills / name, digest)
print(f"Verified {len(provenance['files'])} provenance entries, "
      f"{len(receipt['artifacts'])} EXP-008 artifacts and canonical playbooks")
PY
```

Then run these checks separately, retaining their individual exit codes:

```sh
python -m pytest -q
python tools/check_information_bundle.py
python tools/audit_information_platform.py evidence/EXP-008/attempt-001 --out ../server-audit-001/platform.json
python tools/run_hopfield_information.py verify --output evidence/EXP-008/attempt-001
```

Create `../server-audit-001` first with `mkdir` (no `-p`), or substitute a
fresh path if it exists. The last command is the existing verifier, not a run
or attempt reset. The snapshot has **261 passing local tests**; record actual
server results rather than assuming they match. Preserve any server failures.

**A successful diagnostic exit is not a successful frozen verifier.** The
diagnostic can complete while recording coefficient/feature failures; its
refits use saved features. The original strict Linux verifier has already
failed, and fully regenerated pipeline portability remains unresolved.
Do not relax its tolerances, substitute the diagnostic, retry until lucky,
overwrite receipt files, or restart the consumed attempt.

## Scientific state and exact next work

The objective is to explain what verified shrimp structures mean for neural
behavior and whether their geometry provides practical benefit. The user
wants numerical evidence and clear figures, including failures, not visual
resemblance or an assumed benefit from a striking map.

We reproduced a shrimp-shaped periodic window in a **fixed-weight,
three-neuron continuous-time recurrent tanh network**. The mapped coordinates
are two weights; this is a parameter-plane structure organizing attractors
in neural state space. It is not evidence of verified shrimps in modern
language-model training. The verified core has primitive return count 9,
period 37.63637538 and nontrivial Floquet spectral radius about 0.0650601.

The current 34-page manuscript has 15 empirical figures / 73 panels and ten
numerical tables. Read `paper/neural-shrimp-working-draft.pdf`; supporting
source and hashes are in `paper/` and `paper/provenance.json`.

EXP-008 collected 3,456 previews and 7,008 target/sham records. Geometry lowered
secondary mean candidate-error prediction MSE by 17.17%, but the primary mean
action benefit was only 0.000857628, with descriptive interval
[-0.007701467, 0.006545434]: **unresolved**. Direct minimum-preview selection
had 41.30% lower mean waveform error than the geometry learner at the same
18-preview budget. This is not an established useful geometry controller.

Across 3,504 coarse/fine pairs, the maximum RMSE discrepancy was 0.476851.
Two later Linux saved-feature diagnostics retained all three penalties and
all 288 learned actions, but feature/coefficient tolerance failures remain.
These bounded agreements do not certify every regenerated input or outcome.

Resume in this order:

1. Verify the migration and record the server's platform audit. Read
   `docs/EXP-008-platform-status.md`, the raw/model audits and
   `docs/next-steps.md`. Diagnose arithmetic differences on completely
   regenerated inputs in an explicitly scoped follow-up. Preserve the original
   Linux failures and immutable EXP-008 source, data and attempt.
2. Freeze a numerical refinement packet for the known step-sensitive cases,
   with a complete inclusion census, fixed controls and tighter integration.
   Specify stable fitting (e.g. QR/SVD) and redundant-column treatment
   prospectively. This is development reanalysis of observed cases, not a
   fresh held-out success. Do not change frozen thresholds retroactively.
3. Then freeze the practical experiment: **can geometry choose four useful
   previews out of eighteen before seeing any new previews?** Compare with
   equally budgeted no-geometry and strong simple controls, plus the full-18
   minimum-preview reference. Declare acceptable quality loss and count all
   map/training/diagnostic costs before target outcomes. The existing learner
   uses preview features and cannot be reused unchanged for shortlisting.
4. Only after measured benefit, consider sequential disturbances and one
   concrete trained recurrent-network task. The server's GPU can support that
   later work once the actual implementation and numerical controls are ready.

No successor target experiment has been frozen or run. No current compute
process or rented resource needs to be resumed for this shrimp project.
Paid compute and paid Pro reviews recorded for this project are zero.
Keep the repo private. Ordinary audits/refinements remain authorized; each
paid Pro request needs explicit human approval under canonical integrity
section 7A. RunPod launch needs its separate costed, approved resource plan.

## Read these records rather than reconstructing history

- `docs/next-steps.md`: authoritative next research decisions.
- `docs/EXP-008-results-audit.md` and `docs/EXP-008-model-results-audit.md`:
  independent raw-data and model checks.
- `docs/EXP-008-platform-status.md`, `docs/EXP-008-platform-linux-*.json`,
  `docs/EXP-008-linux-*-failure.txt`: bounded Linux agreements and retained failures.
- `experiments/hopfield/EXP-008-plan.json` and
  `evidence/EXP-008/attempt-001/receipt.json`: frozen protocol and raw bindings.
- `docs/literature/2026-09-11-aguilera-demartino.md`: cautious assessment of
  arXiv:2609.07341v1, an unreviewed multiple-memory hypothesis; its capacity
  simulations are not replicated here. The identity audit identifies missing
  factors in printed equations, not a demonstrated simulation bug.
- Earlier EXP-001--003 training pilots remain negative/unresolved; EXP-004
  establishes the actual recurrent-network shrimp; EXP-005--007 develop
  attraction and waveform-control evidence. Retain their original results,
  failed interpretations and continuation tolerance failure.

The handoff is complete only after the server paths, repository revision,
checks and a resumed chat reading this packet have been confirmed. Keep the
old machine's working copy and conversation until then.
