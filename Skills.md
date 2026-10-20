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
not a paid-cloud escalation. Since 6 October 2026, runs follow the
[adaptive research protocol](docs/adaptive-research-protocol.md): exploratory
runs need committed code and a research-log run card; frozen designs precede
outcomes only for confirmatory tests of headline claims.

Observed on `zeta` on 14 September 2026:

| Resource | Observed capacity |
|---|---|
| CPU | Intel Core i9-14900K; 24 cores, 32 logical CPUs |
| System memory | 96,204,544 KiB reported by Linux, about 91.75 GiB usable RAM |
| GPU 0 | NVIDIA RTX 2000 Ada Generation; 16,380 MiB reported VRAM |
| NVIDIA driver | 580.178.04; `nvidia-smi` succeeds on the host |
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
  provide the basic inventory. A restricted-environment driver-access failure
  is not proof that the host GPU is unavailable; use an available authorized
  host query, and otherwise record availability as unknown.
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
analysis. An optional float64 CUDA RK4 backend is now implemented and tested
on this server. EXP-012 completed all 54 planned batches on 15 September 2026:
the intended full-batch pipeline took median 69.69 seconds on CPU and 75.86
seconds with GPU integration. It also failed the frozen cross-device state
and metric tolerances, although tested candidate rankings were unchanged.
Keep the existing scientific CPU path as the default; the CUDA path remains
an engineering candidate, not a qualified replacement. See
[EXP-012 results](docs/EXP-012-results-audit.md). This outcome does not prohibit
GPU use for other suitable, separately validated workloads.

For an accelerator implementation, first profile and time a representative
unit, then validate against the CPU reference with the same scientific
inputs, precision, stopping rules and diagnostics. Preserve full-state
closure, primitive-period, stability and boundary checks. Record the device,
driver, library versions, arithmetic settings, thread counts, timings and
peak memory. Do not silently lower precision or relax numerical thresholds
to obtain throughput or make the existing EXP-008 replay pass.

Use local CPU and RAM for the continuing fresh-origin research and saved
evidence analysis. EXP-012 identifies CPU waveform scoring and verification
as the dominant cost in the tested GPU preview pipeline; kernel timing alone
is not a pipeline speedup. The server migration does not resolve the retained original strict
Linux replay failure. Preserve that evidence and require a new reviewed
freeze before changing numerical methods or execution defaults.

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
Paid review calls remain separately controlled by
AGENTS.md and the canonical research-integrity playbook.

## Record spend only in the private cost ledger

Human directive, 7 October 2026: dollar amounts actually spent (paid
consults, rented compute, balances, rates paid) are recorded **only** in the
gitignored `cost-ledger.md` at the repository root, never in committed files
(docs, research log, approvals, receipts, commit messages). Committed records
keep the non-monetary facts: approval references, scope, model or GPU, times,
token or hour counts, and hashes, with "cost: see private ledger". Budget
requests and ceilings are discussed with the human in conversation and noted
in the private ledger. The pod ledger in the data store (`runpod/pods.jsonl`)
is outside Git and may hold rates. Each rented GPU run is still listed in the
committed `docs/hardware-ledger.md` with its exact GPU model, driver, CUDA
build and library hash, so results can be reproduced on similar hardware.

## Checkpoint code regularly; keep bulky data outside Git

The user requested regular Git backups on 14 September 2026. Commit and push
to the private origin after each coherent work unit, approximately every
30 minutes during sustained work, and before a long run, handoff or stopping
point. Verify that the remote branch resolves to the pushed commit. Do not
leave completed code only in the working tree or in a local commit. If a
push fails, preserve the local commit, record the failure and retry after
fixing access; do not report a remote backup until it is verified.

Commit on `main` (see AGENTS.md: short-lived branches only when parallel
agents need isolation, merged back promptly; no pull requests required during
the early research stage). Stage files explicitly, inspect the staged diff,
sweep for secrets and inspect staged file sizes before committing, and run the
applicable tests first. A partial implementation may be checkpointed with its
unfinished/test status clearly recorded. A confirmatory freeze still requires
the complete plan and executable code, local review and applicable tests
before the pre-outcome push; exploratory runs follow the adaptive research
protocol. Never force-push main or add assistant coauthor trailers.

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

## Temporary hosted-CI pause

On 22 September 2026 the user requested pausing or reducing GitHub Actions
after billing failures. The research branch now schedules tests only through
manual dispatch; preserve that setting until the user requests resumption.
Continue appropriate local tests and scientific verification on this server.
This scheduling change does not relax any numerical qualification gate or
retroactively pass a failed historical check. The protected-main change is
prepared separately; see [CI pause status](docs/ci-pause-2026-09-22.md).

## Human-friendly research summaries

The user requested on 17 September 2026 that every research summary include a
plain-language TLDR answering: **What did we learn about the behavior of neural
networks in regimes of the shrimp?** Lead with the behavioral consequence
(memory, forgetting, sensitivity, stability, readout or control), then distinguish
what was measured, what is a conditional prediction and what remains unknown.
State the tested network's scope and avoid implying trained-network or
shrimp-specific benefit from generic oscillator behavior. Technical audit and
experiment counts support the explanation; they do not replace this TLDR.
