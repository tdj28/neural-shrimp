# EXP-006 independent utility audit

Read-only recomputation from the complete saved attempt; no target integration or policy selection.
Source: `ddb568e6cd67ef2cfbdb45e05710cd86025ef548`. All 4472 receipt artifact byte counts and SHA-256 values match.
Census: 4320 / 4,320 records; 12 / 12 anchors eligible. Statuses: {'complete': 4320}.
Replayed 4320 saved activation waveforms at their recorded one global phase; maximum RMS discrepancy 0.
Receipt wall time before final verification: 423.866 s.

The pooled primary contrast does **not resolve an overall advantage over the
strong local comparator**: its descriptive interval includes zero. The local
policy has lower error at every anchor for each of the two smallest doses.
At the largest fixed dose, the atlas has 35.35% lower mean error than the local
policy, with a positive descriptive interval and eight of twelve positive
anchor contrasts. The effect changes with dose; the atlas is not uniformly
better. Its pooled contrasts against random and shuffled directions are
positive at every anchor. Neither comparison charges a matched total budget.

## Primary step 0.005: new waveform behavior

Positive benefit means comparator RMSE minus atlas RMSE. Intervals resample the 12 fixed eligible anchors; they are descriptive, not population confidence statements.

| Radius | Atlas RMSE | Local symmetric RMSE | Random symmetric RMSE | Shuffled inward RMSE | Atlas nine-peak retention |
|---:|---:|---:|---:|---:|---:|
| 0.00025 | 0.010773101 | 0.000421635 | 0.008905115 | 0.016758334 | 48/48 |
| 0.001 | 0.036397254 | 0.001724769 | 0.062728596 | 0.190021020 | 44/48 |
| 0.004 | 0.072169791 | 0.055164527 | 0.401483286 | 0.460821585 | 48/48 |
| 0.016 | 0.301671822 | 0.466652880 | 0.741289745 | 0.735004335 | 48/48 |

| Radius | Comparator | Benefit | Descriptive 95% interval | Positive anchors |
|---:|---|---:|---|---:|
| 0.00025 | local_symmetric | -0.010351466 | [-0.014902359, -0.006085704] | 0/12 |
| 0.00025 | random_symmetric | -0.001867986 | [-0.004683145, +0.000660859] | 5/12 |
| 0.00025 | shuffled_inward | +0.005985233 | [+0.001735065, +0.011679185] | 11/12 |
| 0.001 | local_symmetric | -0.034672485 | [-0.056315050, -0.015611397] | 0/12 |
| 0.001 | random_symmetric | +0.026331343 | [-0.009432706, +0.084951525] | 7/12 |
| 0.001 | shuffled_inward | +0.153623766 | [+0.015423171, +0.316870077] | 10/12 |
| 0.004 | local_symmetric | -0.017005264 | [-0.093829363, +0.069591435] | 3/12 |
| 0.004 | random_symmetric | +0.329313495 | [+0.198859696, +0.457963589] | 11/12 |
| 0.004 | shuffled_inward | +0.388651794 | [+0.244674570, +0.536482732] | 12/12 |
| 0.016 | local_symmetric | +0.164981057 | [+0.013853247, +0.326023286] | 8/12 |
| 0.016 | random_symmetric | +0.439617923 | [+0.271913725, +0.592656917] | 11/12 |
| 0.016 | shuffled_inward | +0.433332513 | [+0.267425137, +0.586449865] | 11/12 |

## Across-radius primary contrasts

- local_symmetric: +0.025737960 [-0.027861630, +0.083080991]; 7/12 anchor-average effects positive.
- random_symmetric: +0.198348694 [+0.148078646, +0.245104317]; 12/12 anchor-average effects positive.
- shuffled_inward: +0.245398326 [+0.184833216, +0.304100796]; 12/12 anchor-average effects positive.

## Eventual period retention is not immediate waveform preservation

The nine-peak label is measured after the independent 10,000-unit burn and
2,000-unit measurement. Waveform error is measured immediately over the first
300 units following the jump. These are different horizons. High-error traces
that eventually return to nine peaks are not thereby certified as periodic
throughout their immediate probe. Transients, phase displacement, changed
waveform and frequency can all contribute; this experiment does not isolate
their individual contributions.

| Policy | Nine-peak retained | Mean waveform RMSE | Mean RMSE if nine retained | Range if nine retained |
|---|---:|---:|---:|---|
| atlas_margin | 188/192 | 0.105252992 | 0.107310071 | [0.000491420, 0.875089785] |
| local_symmetric | 308/384 | 0.130990953 | 0.033886543 | [0.000099850, 0.840058844] |
| random_symmetric | 224/384 | 0.303601686 | 0.078029548 | [0.000702922, 0.853588203] |
| shuffled_inward | 100/192 | 0.350651318 | 0.109003618 | [0.002361462, 0.871369324] |
| inward | 112/192 | 0.363674745 | 0.110943864 | [0.002361462, 0.871369324] |

## Numerical refinement and zero-dose floors

- dt=0.01: zero-dose mean/max 6.51925444574e-05 / 6.97642254178e-05; 0 cadence flags.
  - local_symmetric: +0.030132907 [-0.024242445, +0.087820649].
  - random_symmetric: +0.199687501 [+0.148554114, +0.248767260].
  - shuffled_inward: +0.248674106 [+0.186275576, +0.308529191].
- dt=0.005: zero-dose mean/max 6.51903756373e-05 / 6.97569958853e-05; 0 cadence flags.
  - local_symmetric: +0.025737960 [-0.027861630, +0.083080991].
  - random_symmetric: +0.198348694 [+0.148078646, +0.245104317].
  - shuffled_inward: +0.245398326 [+0.184833216, +0.304100796].

Cross-step paired diagnostics:

All paired runs agree on eventual nine-peak retention. Aggregate policy
contrasts have the same qualitative dose pattern at both steps, but immediate
waveform errors are not uniformly converged case by case: their maximum
cross-step difference is 0.3451. Do not describe every trajectory or score as
numerically converged merely because the aggregate result agrees.

```json
{
  "paired_trials": 2160,
  "same_period_count": 2158,
  "same_nine_retention_count": 2160,
  "rmse_absolute_difference_mean": 0.015502957970416406,
  "median": 3.568774350612078e-07,
  "p95": 0.09141699826090378,
  "maximum": 0.34511102701493546
}
```

## Frozen illustrative example

```json
{
  "atlas_margin": {
    "error": 0.09455876894184022,
    "period": 9
  },
  "local_plus": {
    "error": 0.0119170429547904,
    "period": 9
  },
  "local_minus": {
    "error": 0.013724779192772554,
    "period": 9
  },
  "random_plus": {
    "error": 0.8220927008212131,
    "period": 9
  },
  "anchor": "anchor-08"
}
```

## Interpretation boundary

The task is fixed-waveform preservation after a held finite weight change. A stable orbit with the same peak count may have a different shape or physical frequency and therefore substantial task error. These observations do not establish a net speedup, a task-agnostic recommendation to seek or avoid shrimps, or transfer to a trained/reasoning model. The local policy and atlas have different information costs; neither is free. All primary contrasts were recomputed without importing the experiment aggregation helper.
