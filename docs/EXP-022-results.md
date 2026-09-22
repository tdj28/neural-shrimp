# EXP-022: historical controls unavailable in the fixed bins

21 September 2026. Historical metadata preparation completed; proposed neural reference acquisition not launched.

**Human-friendly TLDR:** We have not learned a new memory result here. The existing near-edge network retains some earlier inputs longer than the core network, but the sampled core path does not provide the controls needed to compare them at matched contraction. The core path does cross both desired contraction levels between saved nodes. This is a sampling gap, not evidence that comparable cycles cannot exist, and it gives us a precise next experiment.

The previously committed six-slot design required two anchors and one candidate from each of two paths near each dominant transverse multiplier target, 0.25 and 0.50. Its exact historical selector found four available slots:

| Planned slot | Historical result | Dominant multiplier |
|---|---|---:|
| Core anchor | Available EXP-018 seed | Not a selection criterion |
| Near-edge anchor | Available EXP-018 seed | Not a selection criterion |
| Core, target 0.25 | No eligible saved node within 0.025 | Nearest: 0.2052939379 |
| Core, target 0.50 | No eligible saved node within 0.025 | Nearest: 0.6016586017 |
| Near-edge, target 0.25 | Original node 17 | 0.2434521884 |
| Near-edge, target 0.50 | Original node 29 | 0.4886399723 |

These are historical seed-admission results, not newly qualified references. Full-panel matching is unavailable. All 114 proposed jobs and 528 nested IVP slots remain unlaunched; there were no new neural integrations, finite writes or decoder probabilities. No numerical execution plan or source freeze was completed for this proposal, and the available subset was not substituted for its six-slot estimand.

The core branch's initial attracting prefix has 18 eligible nodes, indices 0–17. Node 18 terminates it with instability and recorded plus-one/unit-circle brackets. The two targets lie between adjacent eligible nodes: 0.25 between nodes 14–15 (multipliers 0.2052939379 and 0.3717062820), and 0.50 between nodes 15–16 (0.3717062820 and 0.6016586017). Both nearest saved distances exceed the fixed 0.025 caliper. Extending the prefix or widening that caliper would change this result's question.

Source preparation checked 51 receipt, hash, journal, deterministic-selection and source-stability conditions. A separately written root audit reproduced the literal selection and gaps in 132 checks without importing the selector. Shared historical inputs mean these are local integrity checks, not independent scientific replication. The [compact receipt](EXP-022-summary.json) locates and binds the complete external records and code.

Reusable six-cycle acquisition and evaluation adapters pass **129 synthetic tests**, including an independently authored analytic nine-maximum oscillator crossing the real saved-packet interfaces. An initial end-to-end analytic test exposed a missing evaluator helper; the failed validation attempt is retained. These engineering checks do not count as new neural evidence.

The prospective successor will test two fixed intermediate core gamma values derived only from these old unforced brackets, retaining the other four seed slots. Moving a parameter requires a bounded seed correction before the unchanged tight same-parameter refinement. The successor must freeze its exact correction method, source, full job census and resource limits before any new integration. A bin miss or numerical failure will remain a failure; there is no memory-guided replacement search. EXP-022 remains closed with its original missing slots.
