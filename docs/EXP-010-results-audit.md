# EXP-010 short-preview numerical qualification: results audit

14 September 2026, project local time. Execution ran on 15 September UTC.
This is a post-run numerical study of previously observed EXP-008 previews.
Its code, complete census and selection of independent sentinels were frozen
and privately pushed before the new integrations.

**Result:** all 3,456 original 75-unit preview labels pass the declared native
scalar and waveform checks. All 16 preselected independent-solver sentinels
also agree in scalar error and waveform. Every candidate ranking, minimum-error
choice and top-four set is unchanged across all 192 contexts and all three
old/refined comparisons. These results support using this complete preview
set for the separately designed retrospective shortlist feasibility study.
They establish no four-preview utility or compute-saving result, and do not
resolve EXP-009's longer, 300-unit candidate outcomes.

The [compact numerical summary](EXP-010-summary.json) retains source and raw
artifact hashes, exact denominators, group diagnostics, the independent panel
and ranking consequences. Full raw data remain outside Git at
`/data2/neural-shrimp-data/raw/EXP-010/attempt-001/`.

## Source, execution and independent evidence checks

The source freeze is `98632899646f3eefa292b8a9f4c1b5123ae37ba0`.
The runtime records live verification of the private branch HEAD at
`2026-09-15T00:51:22.357802+00:00`, before execution began at 00:51:22 UTC.
The receipt was sealed at 00:53:42 UTC. Its 139.8876630510058 seconds include
building, collection, summary and receipt preparation; they exclude the
subsequent saved-array verification. The producer exited zero after that
verification passed. This is not an end-to-end performance benchmark.

| External artifact | Bytes | SHA-256 |
|---|---:|---|
| `receipt.json` | 2,657,369 | `f07aa1cfbb1c4dd5fcaed57d5262ff67f4141fff55adad0a4e690b835d4b651f` |
| `manifest.json` | 2,113,518 | `35d41105dacf322e1c8f9fabfdbe326beae426f31c094edfc93eade3d4a6e405` |
| `summary.json` | 6,314,565 | `66579cc50324ac954bcf1283d1676553e3bcd743884a75e9328f3f5778c5ac7b` |
| `terminal-records.json` | 26,391,741 | `e1d77dd7c8b2b84804095a8b1c2bfa33cee4183988d362fe92271dcf11a1ccfd` |

The independent audit rehashed every one of the **13,896 receipt-listed
artifacts**, totaling **219,453,465 bytes** excluding the receipt itself.
It checked the exact directory census, all **3,463 bound old-input files**
against their original receipt, all **14 bound source files** against the
named Git commit, and the committed manifest bytes. All bindings match.

The audit reconstructed the complete original key product and sentinel
selection from the old journals: 24 origins, two phases, four radii and 18
ordered candidate entries, with 1,728 previews in each original role. Each of
the eight role/radius strata contains 432 entries; its maximum and lower
median at zero-based index 215 give the same 16 distinct sentinels. Duplicate
physical directions retain separate candidate IDs and integrations.

All **6,944 required traces** are complete: 6,912 native traces and 32
independent traces. There are no failures, timeouts, missing records,
substitutions or restarted attempts. Terminal records match the append-only
journal and individual record files. The audit checked each record's planned
task, method and distinct raw path, solver specification, exact original
initial state, destination and control delta, all 751 prescribed sample times,
and the complete finite state array. All 6,944 recorded worker PIDs were gone
when independently checked after collection; no process was terminated by
the audit.

For every trace, a separate NumPy linear-interpolation implementation rebuilt
the endpoint-excluded reference at the saved single global phase. It
recomputed aggregate, per-component, synchronous and window RMSE, and error
at the old preview's saved phase. All **17,360 windows** were included.
The largest metric/window discrepancy was **4.440892098500626e-16**; the
largest old-fixed-phase discrepancy was **1.1102230246251565e-16**.
All **3,504 native/independent waveform-pair comparison trees** recomputed
exactly. Every comparison passes the unchanged arithmetic rule
`rtol=1e-12, atol=1e-14`.

A second independent reconstruction, importing no producer source, passed
56,275 exact equality checks on terminal-value scalar gates, original
assignments, all 12 role/radius/subset groups, all independent sentinels and
every context score, ranking and comparison field. Its waveform qualification
uses the comparison values independently reconstructed from raw arrays above.
The compact summary preserves both audit reports and their script hashes.
Verified durable copies of both independent scripts and reports are stored
separately from the immutable raw attempt at
`/data2/neural-shrimp-data/audits/EXP-010/post-run-audit/`; their exact names,
byte sizes and SHA-256 hashes are retained in the compact summary.

This independent audit did not optimize a new phase or integrate dynamics.
It verifies the recorded objective at its saved phase. The producer's
separate full saved-array replay checks the unchanged phase-search and
cadence metadata as well. Neither check proves an exact global phase minimum,
certifies the original reference orbits again, or establishes general
cross-platform reproducibility. The original strict EXP-008 gate is unchanged.

## Label and waveform qualification

For each preview, the declared native scalar diagnostic requires both
`d2=abs(RMSE_0.0025-RMSE_0.00125)<=1e-4` and
`d2<=d1=abs(old_RMSE_0.005-RMSE_0.0025)`.
All 3,456 cases pass both gates; none is unresolved or undefined. Every case
also passes the separate whole-preview activation waveform thresholds of
RMS difference <=1e-3 and maximum component difference <=1e-2 at equal times.
These thresholds describe numerical agreement at the tested resolutions,
not acceptable task loss or an error bound on the exact solution.

| Original role | Radius | Cases, all passing | Maximum native `d2` | Maximum old-to-finest RMSE change |
|---|---:|---:|---:|---:|
| Development | 0.00025 | 432 | 1.102828e-9 | 1.901614e-8 |
| Development | 0.001 | 432 | 6.567336e-8 | 1.129592e-6 |
| Development | 0.004 | 432 | 1.400909e-7 | 2.415460e-6 |
| Development | 0.016 | 432 | 5.818072e-6 | 9.833794e-5 |
| Test | 0.00025 | 432 | 6.906609e-10 | 1.183349e-8 |
| Test | 0.001 | 432 | 2.059372e-8 | 3.562240e-7 |
| Test | 0.004 | 432 | 4.633375e-7 | 7.975119e-6 |
| Test | 0.016 | 432 | 1.106232e-6 | 1.888737e-5 |

The largest native scalar difference, 5.81807222643782e-6, occurs at
`development-11-p675-r0.016-c06`. That case also has the largest
old-to-finest absolute score change, 9.833793887181752e-5. The largest native
activation RMS discrepancy is 5.251209081587089e-5; the largest activation
component discrepancy is 9.264005489355367e-4. These pass the waveform gates.

All 220 unique previews corresponding to the old policies' 384 test
assignments pass. Their radius counts are 73, 61, 47 and 39 in increasing
radius order. Their largest native `d2` is 5.582166395790011e-10, and
the largest old-to-finest change is 9.572899674348267e-9. These are an
overlapping diagnostic subset of the full census;
no policy was refitted or given a new choice here.

All 16 sentinels pass all three scalar comparisons among finest RK4 and both
DOP853 settings, and all corresponding waveform comparisons. The largest
independent pairwise RMSE difference is **4.416427223929986e-10**. Maximum
independent activation RMS and component differences are
**2.107540779348952e-8** and **2.445070438339414e-7**. The panel is a
prospectively fixed diagnostic selection from old scores, not a random sample
or an independent-solver test of every candidate.

## Rankings and minimum-error consequences

For each of the 192 contexts, all 18 candidate scores were available at the
old 0.005 step and both finer steps. Complete rankings, argmins, top-four
membership and signed-axis argmins are unchanged in every comparison:
old-to-0.0025, old-to-0.00125 and 0.0025-to-0.00125. Candidate-index tie breaks
and duplicate entries remain intact. No favorable context or radius was
selected after observing these results.

The largest old-to-finest change in the minimum over all 18 candidates is
**3.1232610914200265e-8**, at development-05, phase 675, radius 0.016. This is
much smaller than the largest individual candidate-score change. An unstable
poor candidate and an unstable minimum would have different implications;
the full ranking and minimum diagnostics were both checked here.

For old-to-finest comparisons, the old best/second gap exceeds twice the
observed maximum candidate-score change in 191 of 192 contexts; the
fourth/fifth gap does so in all 192. Both gap checks pass in all 192 contexts
for the pair of finer steps. Actual rankings are unchanged even in the
single context lacking the stronger old-to-finest gap separation:
development-04, phase 675, radius 0.004. There, the old best/second gap is
5.170518583633155e-7 and the observed old-to-finest envelope is
1.5301601101191942e-6; candidate 16 remains the winner, and minimum error
changes by only 8.058153484757913e-10. The same context also lacks the gap
separation for old-to-0.0025. These are empirical comparisons between
evaluated numerical levels, not bounds on
unknown exact-solution scores or a task-quality noninferiority margin.

## Interpretation and next use

The complete short-preview data qualify under this frozen numerical protocol.
That removes the preview-label prerequisite for a bounded retrospective
feasibility comparison using the
[pre-existing shortlist design](shortlist-feasibility-design.md). All old
EXP-008 origins are development knowledge for that next study, and EXP-007
reuses twelve of them. Candidate selection must precede preview access, with
identical allowed inputs, matched four-preview controls, and the complete
18-preview reference.

No shortlist has been learned or evaluated by EXP-010. Its stable 75-unit
rankings do not show that four previews preserve a later 300-unit waveform,
that geometry adds useful information, or that total compute is reduced.
EXP-009's 953 unresolved long-target cases, the original EXP-008 coefficient
and arithmetic failures, and the distinction between prediction and action
utility all remain on record. Any subsequent fit or utility result needs its
own pushed executable freeze and complete cost accounting.

The integrated source freeze passed 348 tests before execution. Local
independent code review closed task/raw identity, solver-metadata and
mandatory-summary verification gaps before outcomes. This results audit is
read-only and used no new simulations, GPU resources, paid compute or paid
review.
