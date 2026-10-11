# Compute plan: local server first, paid resources after a costed decision

Status: an offline packaging and workload-control implementation, not a provisioned RunPod experiment. No paid resource or paid review has been requested by this plan. Current NumPy/SciPy experiments run on CPU. No CUDA code, GPU parity, remote throughput, remote resumption, or GPU speedup has been demonstrated. Buying a GPU for the current runner would not accelerate its Python/NumPy code automatically.

The governing local playbook is [`GPU_COMPUTE_SKILLS.md`](../../agent-skill-documents/GPU_COMPUTE_SKILLS.md), especially §§1–4 and 6–7. Its §3 requires an explicit, costed go-ahead for each paid run. Repository creation, general research authorization, and a standing budget do not supply that approval. Routine experiments and audits remain local until the next rung is justified.

## Server amendment, 14 September 2026

The project now runs on `zeta`: an Intel Core i9-14900K (24 cores, 32 logical
CPUs), about 91.75 GiB usable RAM, and an NVIDIA RTX 2000 Ada Generation GPU
with 16,380 MiB reported VRAM. The user authorizes use of all three local
resources. Follow [Skills.md](../Skills.md) for the measured inventory,
availability checks and coordination with other agents. The GPU is shared;
its full memory and compute capacity must not be assumed available.

Prefer the server CPU for the existing reference runners and use the local
GPU when a suitable accelerator path has been implemented, numerically
validated and timed. RunPod remains acceptable when the local GPU is
insufficient or occupied; a CPU/RAM bottleneck can justify a CPU-only
alternative. Prepare a measured, costed run packet and obtain the required
hard-ceiling approval before a paid launch. The existing packaging and
provider-lifecycle limitations below still apply. Historical timings are
observations from their original machines, not benchmarks of this server.

## What is implemented

`tools/prepare_runpod.py` uses only the Python standard library and makes **no network or provider calls**. It provides:

- `bundle`: export allowed committed source/configuration files, a committed manifest, and explicitly selected committed inputs from an exact clean Git commit. Produce an uncompressed deterministic tar archive and report its SHA256, source commit, manifest SHA256 and receipt SHA256. Reject symlinks, submodules in selected paths, unsafe paths, credential-like names and common embedded key formats. Ignored `.env` files, local virtual environments, `.git`, personal credentials and untracked files cannot be copied by this source export. The content check is a guard, not a complete secret detector; manually audit the exact selected release too.
- `verify`: check the externally supplied receipt hash and commit, every file's SHA256 and byte count, absence of symlinks, and absence of unlisted files in an already unpacked payload.
- `estimate`: calculate base, slow-case and hard-ceiling costs from a measured unit time and an explicitly supplied current quote. Refuse non-finite/negative values, an unaffordable hard ceiling, a slow case exceeding the time ceiling, or a no-progress timeout shorter than a slow unit. There are no built-in cloud prices.
- `run`: on an already approved task-owned pod, verify the source, cost and ownership receipts; invoke `python -m neural_shrimp.cli run --manifest ... --output ...`; enforce a compute deadline and an artifact-progress timeout; kill the workload process group when needed; preserve stdout, stderr, raw result files, execution metadata and a SHA256 ledger in an exclusive attempt directory. A used attempt ID cannot be restarted or overwritten by this runner, including after failure.

`tools/runpod_entrypoint.sh` invokes the last command. It neither installs dependencies nor obtains credentials. Its output is a completed process receipt, **not** a scientific validation result and **not** proof of pod deletion. The wrapper does not implement provider provisioning, account inventory, remote retrieval, billing metering, a controller watchdog, or provider teardown. Those operations remain a pre-launch checklist and must be implemented/qualified or conducted by the authorized operator before paid use. A local timeout alone does not stop RunPod billing.

## Cheap-node ladder and scientific gates

1. Run the CPU pilot and unit tests locally. Freeze and publish the exact private-repository commit and manifest before consuming the experiment attempt. Preserve failed attempts.
2. Measure a representative completed parameter-grid unit after warmup with the same iteration limits, dtype, model size, diagnostics and raw-output settings. Include setup/warmup and retrieval separately. The CLI receipt and unit timing are evidence, not an assumed accelerator speed factor.
3. Profile before scaling. Candidate follow-up work is batched/vectorized CPU parameter scans. A GPU successor is conditional on an implemented accelerator kernel, a measured bottleneck it addresses, and float64/parity validation around periodic-window boundaries. Qualify it on the local GPU when available and suitable. Changing dtype or arithmetic creates a new numerical contract.
4. If local runtime, RAM, GPU capacity or contention justifies a cloud run, compare current CPU and GPU total costs using provider quotes and measured relevant throughput. A CPU-only instance is the natural first candidate for this reference code. Local timing is provisional for remote planning; a target-machine qualification is still necessary.
5. Record a base estimate, plausible slow case, cumulative spend, hard lifetime/spend ceiling, artifact-progress timeout, output-volume estimate and retrieval reserve. Obtain explicit approval for that particular paid run, including any bounded qualification phase. No automatic retries or successor resources are authorized.

The training CLI consumes locally prepared, SHA256-bound synthetic input `.npz` files declared by seed in the manifest. Prepare these with the CLI `prepare-inputs` command before freezing the commit; retain the exact input files under `experiments/inputs/`, which the source bundle includes. The execution path verifies hashes and shapes and saves input copies in its results. It does not rebuild training datasets on the pod.

The current implementation has no verified resume path. Use on-demand capacity for any initial paid confirmatory run. Spot/interruptible capacity is excluded until deterministic shard recovery survives a local forced-kill/restart test and restores the complete scientific state without duplicate units.

## Prepare a reviewable payload offline

Commit the runner, lock/environment specification, manifest and any required numerical input artifacts. The manifest belongs under `experiments/manifests/`. The bundle includes complete committed source trees under `src/` or `neural_shrimp/`, `tools/`, `experiments/`, `configs/` and `schemas/`, plus recognized root project requirements files. Additional frozen inputs can be selected using repeated `--input` arguments. Inputs must already be tracked at the frozen commit; the command never scoops arbitrary local data into an archive.

From the repository root, after checking that `git status --porcelain --untracked-files=all` is empty:

```bash
python tools/prepare_runpod.py bundle \
  --repo . \
  --manifest experiments/manifests/EXP-002-training-pilot.json \
  --output /tmp/neural-shrimp-source.tar
```

Save the printed archive SHA256, full source commit and receipt SHA256 on the trusted controller. The archive has a single `payload/` root. Its `runpod-bundle.json` contains `source_commit`, `source_clean`, `manifest`, `inputs`, and a `files` map keyed by relative path with `sha256`, `bytes` and `mode`. The CLI can obtain source provenance from this receipt when `.git` is absent. The external controller's receipt hash is the integrity anchor; trusting a receipt and source that were both replaced would not establish provenance.

This is a self-contained **source and frozen-input** payload, not a vendored Python interpreter or dependency bundle. The archive includes `uv.lock`. Qualify a digest-pinned runtime image containing the exact locked NumPy/SciPy/Matplotlib versions in advance, and retain its environment receipt. Build the local qualification environment with `uv sync --frozen --extra dev`; a runtime environment may omit `--extra dev`. Alternatively export locked requirements using `uv export --frozen` and use those in the image build. Complete dependency/image preparation before paid execution. Do not discover imports, install an unpinned latest dependency set or clone a moving Git branch after a costly pod starts. No private GitHub token is needed on the pod when this payload is transferred by the trusted controller.

Do not unpack an untrusted archive using a bare `tar` command. This exporter emits regular files only; verify the archive SHA256 against the local controller receipt before extracting it into an empty directory, then run `verify` with the expected receipt SHA256 and full commit.

## Cost report and approval packet

The estimate interface deliberately has no default price or timing. Run `python tools/prepare_runpod.py estimate --help` for all required arguments. Supply:

| Input | Meaning |
|---|---|
| `--unit-seconds`, `--units` | Measured steady-state seconds per work unit and exact planned unit count |
| `--measurement-ref`, `--measurement-host` | Raw timing receipt/hash and machine description; disclose local-versus-target timing |
| `--hourly-usd`, `--quote-source`, `--quote-observed-at` | Effective current quoted rate, exact provider quote reference, and observation time |
| `--setup-seconds` | Provisioning, transfer, import, loading and warmup allowance |
| `--retrieval-seconds` | Positive reserve for receipt creation, artifact transfer, verification and deletion |
| `--storage-transfer-usd` | Separate conservative allowance, including retained storage charges when relevant |
| `--slowdown-factor` | At least 1; multiplies steady-state compute in the slow case |
| `--hard-wall-seconds`, `--hard-budget-usd` | Approved total resource lifetime and total spend ceiling |
| `--no-progress-seconds` | Frozen maximum time without a changed scientific output artifact |
| `--output` | New cost-report JSON path; existing reports cannot be overwritten |

Base cost is `hourly_usd × (setup_seconds + unit_seconds × units + retrieval_seconds) / 3600 + storage_transfer_usd`. The slow case multiplies only the compute term by `slowdown_factor`. The worst permitted lifetime must also fit the spend ceiling. Storage or network charges beyond the supplied allowance are not magically capped by this arithmetic: the controller must enforce the approved resource configuration and lifetime.

An approval packet must bind the plan, source/manifest/input hashes, cost-report hash, units, measured timings, current quote, actual image digest, expected output volume, maximum spend, prior cumulative spend and a specific human approval reference. The program's `NOT APPROVED` report label is intentional; arithmetic never grants authorization. Do not ask for an ordinary paid Pro research review as a compute prerequisite.

## Identity and execution contract

Before provisioning, an authorized controller records the read-only account inventory and every pre-existing resource ID. Record the exact ID returned by the single create operation for this task. No pre-existing or unfamiliar pod may be inspected over SSH, stopped, restarted, resized or deleted without its owner's specific authorization.

The externally hash-bound ownership receipt requires these fields:

```json
{
  "task_id": "the-approved-task-id",
  "pod_id": "exact-id-returned-by-create",
  "preexisting_resource_ids": [],
  "create_response_sha256": "64-lowercase-hex-digits-of-retained-provider-response",
  "billed_start_epoch": 0,
  "controller_watchdog_ref": "receipt-for-independently-running-controller-watchdog",
  "runtime_image_digest": "registry/image@sha256:64-lowercase-hex-digits"
}
```

This is a schema illustration, not executable authorization. Use the actual inventory, creation response, beginning of billing and observed runtime. The wrapper rejects `billed_start_epoch=0`; the deployment receipt must supply the real time. `RUNPOD_POD_ID` must match the receipt as a cross-check; environment variables alone are not identity proof. The receipt records the controller watchdog, but the wrapper cannot prove that watchdog is alive; validate it independently before launch.

The workload command requires all of `--expected-commit`, `--expected-receipt-sha256`, `--cost-report`, `--expected-cost-sha256`, `--ownership-receipt`, `--expected-ownership-sha256`, `--task-id`, `--attempt-id`, `--approval-ref`, and `--output-base`. The latter must be outside the frozen payload. Use `tools/runpod_entrypoint.sh --help` to inspect the interface. The source archive and output paths must be writable only by the task process/operator; file hashes are provenance checks, not kernel confinement against another writer.

The hard compute deadline is the recorded billing start plus hard lifetime, minus the retrieval reserve and five seconds for process termination. Thus setup consumes the same overall lifetime budget. The artifact-progress watchdog resets only when file name, size or modification time changes under the scientific results directory; stdout chatter does not count. This detects a stalled shard writer, not scientific success, and its interval must exceed the longest allowed work unit. A hung or failing workload preserves the same attempt and cannot silently retry.

The receipt does not dump the environment or credentials. The workload process strips common token/password/API-key environment variables; this is secondary protection. The controller must keep its cloud API key local and must not place any credential, broad private SSH key, model token or `.env` file in the source/archive/result tree. No credentials are needed by the current synthetic experiments.

## Retrieval and termination checklist

Before a paid run, assign the authorized operator/controller responsible for every item. The implementation intentionally cannot claim these as completed:

1. Install and test an **independent controller** hard-deadline watchdog, bound to the provider-issued task pod ID and billing-start time. It must still terminate that pod if the workload or SSH connection dies. Reserve enough time to retrieve partial failures; do not wait for the expensive machine while debugging locally.
2. On completion, failure, timeout or no progress, close the workload and produce its remote `SHA256.json` ledger before transfer. The ledger includes launch/execution metadata, logs, completed raw shards and available partial files. It excludes itself; retain its own SHA256 separately on the controller.
3. Retrieve the entire attempt tree, including failures and partial output, into a new local location. Verify each relative path, byte count and SHA256 against the remote ledger, compare expected/completed trial counts, and preserve the ledger hash and transfer receipt. Do not label incomplete output a successful scientific experiment.
4. Remove temporary workload credentials if any were explicitly approved, verify absence, then immediately terminate the **owned** pod. Do not leave an idle machine to hold a cache.
5. Retain the provider deletion response, directly query the exact pod ID until absent, and reconcile a new full account inventory with the pre-launch inventory. A delete request or a stopped state does not prove zero continuing charges. Separately reconcile any task-created persistent/network volumes; do not mutate unfamiliar resources.
6. Record final actual spend, elapsed billing time, retrieved artifact hashes, absence verification and any persistent-storage disposition in the local run ledger. Keep all durable evidence locally and in the approved private repository/artifact store.

As of documentation checked 2026-09-08, RunPod documents CPU and GPU deployment, continuing volume charges on stopped pods, and distinct termination behavior. Consult the current official [Pod management documentation](https://docs.runpod.io/pods/manage-pods), [pricing documentation](https://docs.runpod.io/pods/pricing), and [delete-Pod API reference](https://docs.runpod.io/api-reference/pods/DELETE/pods/podId) when preparing an actual quote/launch. This plan embeds no provider rate or unverified endpoint command.
