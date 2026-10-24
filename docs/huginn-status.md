# Huginn internal-loop research status

Updated: 2026-10-24. **Paused at the human's request.** This is a restart note, not authorization for more experiments. No Huginn rental remains running for the completed HPC3 sweep; its exact pod was independently confirmed absent. The `finish-huginn-bifurcation-sweep` follow-up is paused. No analysis is intentionally left running. The proposed 64 × 64 map is deferred.

## Where we landed

We observed robust approximate two-step repetition at some modified internal-loop settings, initialization/history dependence, and positive finite-window separation growth elsewhere. We have **not established sustained chaos, a period-doubling cascade, Feigenbaum scaling, verified primitive-orbit stability, or shrimp organization in Huginn**. Nor have we shown that chaos is absent. The narrow transition follow-up does not settle the interpretation of the stronger-growth region in the earlier map: they use different parameter coordinates.

The latest apparent period-48 return is window- and tolerance-sensitive. The best current description of that ascending trajectory is near-two-step repetition with small time-varying deviations. Numerical effects, slow modulation, and long transients remain unresolved. Do not present the lag-48 label as a discovered stable period-48 orbit.

## What system was tested

This is Huginn's recurrent **hidden-state** loop with a fixed prompt, not our external soft-token feedback system. The prompt was `What is 17 times 23? The answer is`. Readouts observe token probabilities; they are not sampled tokens fed back into the model, and alternating top predictions are not generated text.

The modified update is schematically

\[
h_{t+1}=(1-\lambda)h_t+\lambda F_\alpha(h_t;e),
\]

where prompt embeddings/context are held fixed, λ controls update size and α controls an injected internal steering direction. See the implementation for the exact steering injection. The native point is steering zero, λ = 1. These experiments primarily explore modified updates beyond the native point, not ordinary deployed inference behavior.

Pinned model revision: `bb6621b65e90b6a4b9b29ef88dc83866d450470c`. Arithmetic was float32 with TF32 disabled. Float64 reductions of saved float32 states do **not** make the underlying trajectory float64.

## Evidence sequence and limitations

| Study | Outcome / limitation | Entry point |
| --- | --- | --- |
| HCM1 broad 32 × 32 map | Native point was contracting and a fixed-point candidate. Other settings showed recurrence candidates and positive finite-window growth. Full tails were not retained, so its recurrence labels cannot be independently replayed from full histories. | [Map receipt](huginn-chaos-map-receipt.json) |
| HCM2 first zoom | Retrieval failed; only partial evidence survives. Never treat this as a complete map. | [Failure receipt](huginn-chaos-zoom-failure-receipt.json) |
| HCM3 tighter 32 × 32 map | λ = 1.320–1.328, steering −0.110 to −0.100. 775 cells exceeded 0.001 growth per update across both offsets and both measurement halves. Full-state tail recurrence checks found no lag ≤64 under the stated test. This is candidate finite-window expansion, not proven chaos; lack of recurrence alone is not stronger evidence of chaos. Scientific dataset recovered incrementally despite preserved legacy archive failure. | [Tight-zoom receipt](huginn-chaos-tight-zoom-receipt.json) |
| HPR1 period-two readout | Extended the periodic candidate locally. Warm trajectory alternated top predictions ` a` / ` on` for 8192 additional updates; full distributions, not just their top choices, returned approximately every two updates. This is a numerical candidate, not an exact orbit proof. | [Readout receipt](huginn-periodic-readout-receipt.json) |
| HPC1 local continuation | Continued/fresh states behaved differently; no period-four ladder established. | [Pilot receipt](huginn-continuation-pilot-receipt.json) |
| HPC2 original and retry | Both incomplete; retain independently audited recovered subsets. The retry bracketed a continued-state change between λ = 1.280 and 1.285. Do not splice the two attempts into a completed sweep. | [Original failure](huginn-continuation-sweep-failure-receipt.json), [retry failure](huginn-continuation-sweep-retry-failure-receipt.json) |
| HPC3 fixed up/down sweep | All 22 settings retrieved, independently audited, and backed up. No demonstrated doubling cascade. | [Audit receipt](huginn-transition-audit-receipt.json) |
| HPC3 saved closure analysis | Lag-48 classification does not persist across longer observation windows. Local CPU only. | [Closure receipt](huginn-saved-drift-receipt.json) |

## HPC3: exact experiment and results

- Steering: **−0.03741935483870967**.
- Eleven ascending λ values from **1.280 to 1.285**, spacing **0.0005**, followed by the same eleven descending values. Upper endpoint is deliberately repeated.
- At each setting: a continued trajectory and a fresh seed-0 trajectory; **16384 settling + 8192 measured updates**.
- Two shared-base perturbation sizes: **0.001 and 0.0003**. This differs from HCM3's separately rerun bases.
- Source: `e2c94998ec26e6347dc0cd30781e3c1efe5e1cb2`; [plan](huginn-chaos-transition-plan.json).
- Ascending continuation begins from the audited HPC2 retry final state at 1.280. The plan's `anchor_update_size = 1.2532258064516129` is the separate saved-step replay coordinate, not that continued anchor coordinate.
- Every setting had to be retrieved and hash-verified before advancing. First-setting compute-plus-transfer timing passed its fixed deadline gate. All 22 acknowledgments passed.
- Final manifest: **579 files**, SHA-256 `75fe8640d8a2f60ca118ec13d40ba36e2b5c28d932edb191cc5932f4782aa171`.
- Exact pod `h4fdwbaw1d5eok` terminated; independent direct lookup and inventory confirmed absence.

Ascending λ = 1.280 and 1.2805 passed approximate lag-two recurrence at all eight measured checkpoints. At 1.281, seven checkpoints passed lag two and the final checkpoint returned lag 48. Later ascending settings, all descending settings, and fresh-start controls did not meet the same recurrence test through lag 64. Direction dependence over this horizon is **not yet proof of asymptotic hysteresis or coexistence**.

The saved closure analysis checked all measured updates at ascending 1.2805, ascending 1.281 and descending 1.281. With relative Euclidean closure tolerance **3 × 10⁻⁵**, ascending 1.281 had these approximate final-128-position maximum errors:

| Lag | Error | Passes original tolerance? |
| --- | --- | --- |
| 2 | 3.1520 × 10⁻⁵ | No |
| 48 | 2.6944 × 10⁻⁵ | Yes |
| 96 | 3.2608 × 10⁻⁵ | No |

Lag 48 also passes over the final 256 positions, but its maximum error reaches approximately **5.2123 × 10⁻⁵** over the final 1024 and **7.0965 × 10⁻⁵** over the whole measured interval. Raising the final-window tolerance by about 5.1% would make lag two pass. Failure at lag 96 under the same finite tolerance is a persistence warning, not a theorem excluding a nearby periodic orbit. Ascending 1.2805 maintains lag-two closure across the measured interval. Descending 1.281 has much larger return errors.

## Where to find the evidence

All raw arrays and generated figures are outside repositories. Git contains code, plans and compact receipts, not the scientific dataset.

- HPC3 controller root: `/data2/neural-shrimp-data/runpod/huginn-chaos/hpc3-20261023`
- HPC3 raw data: `/data2/neural-shrimp-data/runpod/huginn-chaos/hpc3-20261023/workers/neural-shrimp-hpc3-20261023/20261024T001350Z/incremental/data`
- Full audit: `/data2/neural-shrimp-data/audits/huginn-chaos/hpc3-complete-20261024-01`
- Sweep figure: `hpc3-transition.png` / `.pdf` in that audit directory. Upper panels show only the upper projection branch, not the entire state space.
- Closure analysis: `/data2/neural-shrimp-data/audits/huginn-chaos/hpc3-drift-20261024-01`
- Closure figure: `closure-persistence.png` in that directory; `summary.json` and `drift-*.npz` retain numerical errors.
- Separate physical-disk backup: `/archive/neural-shrimp-backups/20261024T064621Z-huginn-hpc3-complete`
- [Backup receipt](backups/huginn-hpc3-complete-20261024.json): all **621 files / 274603203506 bytes** checked by independent destination and repeated source hashing. `scientific-audit/` and `saved-drift-analysis/` contain separately hash-verified supplemental analysis copies. This is a same-machine separate-disk backup, not offsite storage.
- Earlier raw/audit roots are recorded in the individual receipts above.
- Costs and approvals involving amounts: **ignored private `cost-ledger.md` only**.

Useful code: `tools/huginn_continuation_sweep.py`, `tools/huginn_chaos_map.py`, `tools/huginn_saved_drift.py`, `tools/huginn_evidence_gate.py`, `tools/huginn_incremental_retrieval.py`, `tools/huginn_retrieval_guard.py`. Saved-closure analysis source was committed at `26e08e11`; its completed receipt and findings at `93207e7a`.

## If the human chooses to resume

1. **First use saved data, without new GPU work.** Examine the every-other-update trajectories to remove dominant alternation. Quantify drift or modulation of each parity separately, including dependence on time window and closure tolerance. The completed lag analysis is not a spectral/modulation identification or a monotonic-drift proof.
2. **Prepare a targeted precision and settling test**, at λ = 1.281 with 1.2805 as control, continued and fresh starts. Qualify effective arithmetic through normalization, rotary embeddings and device hooks before claiming improved precision. Measure deterministic numerical repeatability and a practical error floor. Do not infer actual precision from dtype labels.
3. If small deviations persist under qualified precision and longer settling, investigate orbit closure and stability and possible slow modulation. If they shrink, reassess the apparent transition. A verified flip crossing would be needed before treating an observed change as a demonstrated period-doubling bifurcation.
4. Reconsider a denser map only after deciding what diagnostic its colors represent. The stronger-growth HCM3 region is a separate target and needs its own validation; this narrow sweep neither validates nor disproves it.

These are proposals, **not active jobs or standing approval to spend**. Any new paid run, including a retry, needs a concrete measured scope, quote, hard ceiling and fresh human confirmation. Recheck local resource availability and the human's GPU needs first.

## Operational lessons that must survive a restart

Three earlier interruptions lost evidence: HCM2's packing was treated as stalled numerical work; HPC2's phase-first JSON was parsed incorrectly; the HPC2 retry collector exited after transient missing SSH endpoint metadata despite a still-working endpoint. Preserve the failure receipts and partial attempts.

Use the tested per-setting retrieval acknowledgment barrier and final manifest verification. Keep numerical progress, transfer progress and verified completion distinct. Retain the known SSH endpoint only after exact resource identity checks. Collector recovery must remain bounded within the original deadline. Do not terminate early merely because a completed setting is waiting for retrieval, but never extend a hard resource ceiling. No monolithic archive is needed for this path. Re-run the lifecycle fault tests before reusing or modifying the paid path; synthetic tests do not guarantee provider durability. Verify a separate-disk copy before claiming the data backed up.

Current stopping decision: **enough for now; preserve the evidence and leave research paused.**
