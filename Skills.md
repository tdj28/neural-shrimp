# Project skills and compute guidance

Effective 14 September 2026, following the user's migration to the shared
`zeta` server. Read this file alongside [AGENTS.md](AGENTS.md) and the canonical
research playbooks in `../agent-skill-documents/`. This local amendment replaces
assumptions that this project runs on a weak computer or lacks a local GPU.
Historical experiment receipts and playbook hashes remain unchanged.

## Use the server first

The user authorizes this project to use the server's CPU, RAM and GPU for
local research, development, validation and analysis. Use whichever local
resource fits the workload; an available local GPU is an ordinary option,
not a paid-cloud escalation. Research freezes still precede target outcomes.

Observed on `zeta` on 14 September 2026:

| Resource | Observed capacity |
|---|---|
| CPU | Intel Core i9-14900K; 24 cores, 32 logical CPUs |
| System memory | 96,204,544 KiB reported by Linux, about 91.75 GiB usable RAM |
| GPU 0 | NVIDIA RTX 2000 Ada Generation; 16,380 MiB reported VRAM |
| NVIDIA driver | 580.178.04; `nvidia-smi` succeeds outside the sandbox |
| Research checkout | `/data2/neural-shrimp` |
| Project Python | `/data2/neural-shrimp/.venv/bin/python` |

These are capacity observations, not a reservation. At 20:52 UTC, Linux
reported about 47.6 GiB available RAM and the GPU reported 5,987 MiB free
VRAM with other processes active. Recheck availability for every substantial
job. Driver visibility does not establish that this project's numerical
code has a working or validated CUDA implementation.

## Share resources with other agents

- Before a substantial run, inspect CPU load, available RAM, GPU utilization,
  free VRAM and active GPU processes. `uptime`, `free -h` and `nvidia-smi`
  provide the basic inventory. A sandbox driver-access failure is not proof
  that the host GPU is unavailable; use the normal permission mechanism for
  a host query when needed.
- Choose explicit worker, thread, batch and memory limits from the current
  available capacity and a measured representative unit. Leave headroom for
  the operating system, desktop and other tasks. Start conservatively and
  increase concurrency only when measurements justify it.
- Bound BLAS/OpenMP threads, especially when using multiple worker processes,
  so parallel jobs do not each claim all 32 logical CPUs. Saved numerical
  replay has used one thread; retain and record that setting for comparable
  audits rather than assuming a thread-count change is numerically neutral.
- Treat every unfamiliar process as owned by another task. Do not kill,
  suspend, reset the GPU, unload its driver or change system-wide settings to
  obtain capacity. Record this task's process IDs and stop only its own jobs.
- When the GPU is occupied, use suitable CPU work, defer the GPU stage, or
  prepare the RunPod fallback below. An idle utilization reading alone does
  not mean another process's allocated VRAM is available to this task.

## Match compute to the scientific workload

Current reference experiments use native CPU integration and NumPy/SciPy
analysis. Continue to use the server CPU for those paths. GPU hardware alone
does not accelerate them; no project GPU kernel or GPU speedup is verified.

For an accelerator implementation, first profile and time a representative
unit, then validate against the CPU reference with the same scientific
inputs, precision, stopping rules and diagnostics. Preserve full-state
closure, primitive-period, stability and boundary checks. Record the device,
driver, library versions, arithmetic settings, thread counts, timings and
peak memory. Do not silently lower precision or relax numerical thresholds
to obtain throughput or make the existing EXP-008 replay pass.

Use local CPU and RAM immediately for the current numerical-portability audit
and saved-evidence analysis. The server migration does not resolve the
retained strict Linux replay failure; GPU porting is a separate engineering
decision and is not a prerequisite for this audit.

## RunPod remains an acceptable fallback

The user explicitly permits considering RunPod when the local GPU is not
sufficient for the workload or is busy with other agents. Capacity,
supported operations, validated precision and measured runtime all matter;
there is no requirement to force an unsuitable job onto the local GPU.
A CPU-only remote instance can also be considered for a CPU/RAM bottleneck.

For a specific paid launch, retain the existing
[compute-plan requirements](docs/compute-plan.md): measured throughput,
current pricing, a costed plan and approved hard spending/lifetime ceiling,
exact task resource ownership, artifact retrieval with hash verification,
and verified termination. This amendment authorizes the fallback choice;
it supplies neither a dollar ceiling nor a specific paid launch approval.
Prepare the concrete costed packet before requesting any missing approval.
Paid review calls and public release remain separately controlled by
AGENTS.md and the canonical research-integrity playbook.

## Checkpoint code regularly; keep bulky data outside Git

The user requested regular Git backups on 14 September 2026. Commit and push
to the private origin after each coherent work unit, approximately every
30 minutes during sustained work, and before a long run, handoff or stopping
point. Verify that the remote branch resolves to the pushed commit. Do not
leave completed code only in the working tree or in a local commit. If a
push fails, preserve the local commit, record the failure and retry after
fixing access; do not report a remote backup until it is verified.

Use a dedicated branch, stage files explicitly, inspect the staged diff,
sweep for secrets and inspect staged file sizes before committing. A partial
implementation may be checkpointed on its private working branch with its
unfinished/test status clearly recorded. A research execution freeze still
requires the complete plan and executable code, local review and applicable
tests before the pre-outcome push. Use a reviewed PR for behavior changes;
never force-push main or add assistant coauthor trailers.

Store new bulky raw arrays, trajectories, checkpoints and verbose run outputs
in `/data2/neural-shrimp-data/`, the sibling `../neural-shrimp-data/`, rather
than in this repository. Use `raw/<experiment>/<attempt>/` for target data,
`audits/<audit>/<attempt>/` for diagnostic output and `checkpoints/` for
large model state. Create a new attempt directory exclusively, retain failed
and incomplete attempts, and never overwrite old raw evidence.

Keep source, tests, protocols, machine plans, small summaries and compact
artifact receipts in Git. Receipts must identify the external storage root,
relative paths, SHA-256 hashes, byte sizes, exact source commit and plan hash,
with enough instructions to locate and verify the data. Review every staged
file above 5 MiB; bulky raw data and archives belong outside Git. The size
threshold is a review aid, not permission to commit many small raw files.
Use the directory exclusions in `.gitignore` if temporary raw output must
exist inside the checkout; do not override them casually with `git add -f`.

Existing committed frozen evidence stays intact. Do not remove it from
history, relocate it without a separate verified migration, or break its
source and manifest bindings. A sibling data directory is durable local
storage, not an independent backup: record any verified additional data
copy separately. Git pushes protect the code and receipts, not external raw
data. See [data-storage.md](docs/data-storage.md).
